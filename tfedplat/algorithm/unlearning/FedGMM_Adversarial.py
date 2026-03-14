import tfedplat as fp
import math
import time
import torch
import torch.nn.functional as F


class AdversarialForgettingLoss:
    """
    Adversarial Forgetting Loss with adaptive alpha-blending.

    L = alpha * L_adv + (1 - alpha) * L_KL

    where:
        L_adv = -log(1 - p_y + eps)          pushes correct-class probability to zero
        L_KL  = KL(softmax(z/tau) || Uniform) prevents distribution collapse

    alpha is dynamically controlled by the algorithm via cosine annealing:
        Early training: alpha ~ alpha_init (lower) -> more KL regularization, stable start
        Late training:  alpha ~ alpha_max  (higher) -> strong adversarial pressure

    This resolves the "antagonistic equilibrium" where a fixed KL weight
    counteracts adversarial pressure in late training, causing the ASR bottleneck.
    """

    def __init__(self, temperature=2.0, weight=1.0, eps=1e-8, alpha=0.5):
        self.temperature = temperature
        self.weight = weight
        self.eps = eps
        self.alpha = alpha  # dynamically updated by the algorithm each round

    def __call__(self, pred, target):
        scaled_logits = pred / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs = F.log_softmax(scaled_logits, dim=-1)

        batch_size, num_classes = pred.shape

        # Adversarial component: -log(1 - p_correct + eps)
        # Gradient drives p_correct -> 0, i.e., model becomes "confidently wrong"
        p_correct = probs[torch.arange(batch_size, device=pred.device), target]
        L_adv = -torch.log(1.0 - p_correct + self.eps)

        # KL-to-uniform component: KL(p || U(1/K))
        # Equivalent to log(K) - H(p), so it also maximizes entropy
        # Prevents collapse to a single wrong class
        uniform = torch.full_like(probs, 1.0 / num_classes)
        L_kl = F.kl_div(log_probs, uniform, reduction="none").sum(dim=-1)

        # Adaptive blending
        loss = self.alpha * L_adv + (1.0 - self.alpha) * L_kl
        return self.weight * loss.mean()


class FedGMM_Adversarial(fp.UnlearnAlgorithm):
    """
    Federated Gradient Masking & Modification with Adversarial Forgetting.

    Two algorithmic innovations:

    1. Progressive Masking (Coarse-to-Fine):
       Mask ratio rho(t) follows a cosine schedule from rho_min to rho_max.
       Early rounds use a small mask for precise backdoor targeting while
       protecting shared parameters. Later rounds widen the mask to sweep
       residual backdoor features embedded in deeper layers.

    2. Adaptive Alpha-Annealing:
       The adversarial/KL balance alpha(t) increases from alpha_init to alpha_max.
       Early training uses more KL regularization for stability; late training
       shifts to near-pure adversarial pressure, breaking through the ~8% ASR
       bottleneck caused by KL counteracting the forgetting gradient.

    Both schedules use cosine annealing: smooth S-curve transition that is
    slow at the start and end, fast in the middle.
    """

    def __init__(
        self,
        name="FedGMM_Adversarial",
        data_loader=None,
        module=None,
        device=None,
        train_setting=None,
        client_num=None,
        client_list=None,
        online_client_num=None,
        save_model=False,
        max_comm_round=0,
        max_training_num=0,
        epochs=1,
        save_name=None,
        outFunc=None,
        write_log=True,
        dishonest=None,
        test_conflicts=False,
        params=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            name, data_loader, module, device, train_setting,
            client_num, client_list, online_client_num, save_model,
            max_comm_round, max_training_num, epochs, save_name,
            outFunc, write_log, dishonest, test_conflicts, params,
        )

        p = params if params is not None else {}

        # --- Progressive Masking ---
        base_rho = float(p.get("adv_rho", 0.15))
        rho_min = p.get("adv_rho_min")
        rho_max = p.get("adv_rho_max")
        self.rho_min = float(rho_min) if rho_min is not None else base_rho
        self.rho_max = float(rho_max) if rho_max is not None else base_rho

        self.mask_refresh = int(p.get("adv_mask_refresh", 5))
        self.mask_ema_beta = float(p.get("adv_mask_ema_beta", 0.0))
        self.use_grad_divergence = (
            str(p.get("adv_use_grad_divergence", "False")).lower()
            in ("true", "1", "yes")
        )

        # --- Adaptive Loss Annealing ---
        self.adv_weight = float(p.get("adv_weight", 1.0))
        self.adv_temperature = float(p.get("adv_temperature", 2.0))
        alpha_init = p.get("adv_alpha_init")
        alpha_max = p.get("adv_alpha_max")
        self.alpha_init = float(alpha_init) if alpha_init is not None else 0.7
        self.alpha_max = float(alpha_max) if alpha_max is not None else 0.7

        # --- Internal State ---
        self.mask_dict = {}           # client_id -> binary mask vector
        self.mask_score_ema = {}      # client_id -> EMA-smoothed score vector
        self.model_params = self.module.span_model_params_to_vec()
        self.correct_prob_history = []

    # ==================== Cosine Schedule ====================

    @staticmethod
    def _cosine_schedule(progress, start, end):
        """Smooth S-curve interpolation: slow at endpoints, fast in middle."""
        return start + (end - start) * (1.0 - math.cos(math.pi * progress)) / 2.0

    def _get_progress(self):
        """Training progress ratio in [0, 1]."""
        T = getattr(self, "max_unlearn_round", 0)
        if T <= 0:
            return 0.0
        return min(1.0, self.current_comm_round / T)

    def _get_current_rho(self):
        """Dynamic mask ratio: rho_min -> rho_max over training."""
        return self._cosine_schedule(self._get_progress(), self.rho_min, self.rho_max)

    def _get_current_alpha(self):
        """Dynamic adversarial weight: alpha_init -> alpha_max over training."""
        return self._cosine_schedule(self._get_progress(), self.alpha_init, self.alpha_max)

    # ==================== Gradient Computation ====================

    def _compute_client_gradient(self, client):
        """Compute gradient for a client on its local data."""
        msg = {
            "command": "cal_gradient_loss",
            "epochs": self.epochs,
            "lr": self.lr,
            "target_module": self.module,
        }
        client.get_message(msg)
        return client.get_message({"command": "require_gradient_loss"})["g_local"]

    # ==================== Mask Construction ====================

    def _compute_divergence_score(self, unlearn_client):
        """
        Gradient divergence score per parameter.

        s_i = -g_i^(u) * g_i^(r) + |g_i^(u)|

        High score => forget and retain gradients conflict on this parameter
        => likely backdoor-specific, should be prioritized for unlearning.
        """
        forget_grad = self._compute_client_gradient(unlearn_client)

        retain_grads = [
            self._compute_client_gradient(c)
            for c in self.client_list
            if not getattr(c, "unlearn_flag", False)
        ]

        if not retain_grads:
            score = torch.abs(forget_grad)
        else:
            retain_avg = torch.stack(retain_grads).mean(dim=0)
            score = -forget_grad * retain_avg + torch.abs(forget_grad)

        # Normalize to [0, 1]
        return (score - score.min()) / (score.max() - score.min() + 1e-8)

    def _compute_magnitude_score(self, grad_vec):
        """Gradient magnitude score per parameter (fallback when divergence is off)."""
        score = torch.abs(grad_vec)
        return (score - score.min()) / (score.max() - score.min() + 1e-8)

    def _update_ema(self, client_id, score):
        """Optional EMA smoothing to suppress mask thrashing from gradient noise."""
        beta = self.mask_ema_beta
        if beta <= 0:
            return score
        prev = self.mask_score_ema.get(client_id)
        if prev is None:
            self.mask_score_ema[client_id] = score.detach()
        else:
            self.mask_score_ema[client_id] = beta * prev + (1 - beta) * score.detach()
        return self.mask_score_ema[client_id]

    def _score_to_mask(self, score):
        """Convert continuous score to binary Top-rho(t) mask."""
        rho = self._get_current_rho()
        k = max(1, int(len(score) * rho))
        _, indices = torch.topk(score, k=k, largest=True)
        mask = torch.zeros_like(score, device=self.device)
        mask[indices] = 1.0
        return mask

    def _build_or_refresh_masks(self):
        """Generate masks for all unlearn clients."""
        for client in self.client_list:
            if not getattr(client, "unlearn_flag", False):
                continue

            if self.use_grad_divergence:
                score = self._compute_divergence_score(client)
            else:
                grad = self._compute_client_gradient(client)
                score = self._compute_magnitude_score(grad)

            score = self._update_ema(client.id, score)
            self.mask_dict[client.id] = self._score_to_mask(score)

    def _maybe_refresh_masks(self):
        """Refresh masks periodically or on first call."""
        if not self.mask_dict:
            self._build_or_refresh_masks()
            return
        if self.mask_refresh > 0 and self.current_comm_round > 0:
            if self.current_comm_round % self.mask_refresh == 0:
                self._build_or_refresh_masks()

    # ==================== Aggregation ====================

    def _aggregate_masked_gradients(self, g_locals):
        """Apply masks to unlearn clients, then weighted-average all gradients."""
        weights = (
            torch.Tensor(self.get_clinet_attr("local_training_number"))
            .float()
            .to(self.device)
        )
        weights = weights / weights.sum()

        masked = []
        for idx, client in enumerate(self.online_client_list):
            grad = g_locals[idx]
            if getattr(client, "unlearn_flag", False):
                mask = self.mask_dict.get(client.id)
                if mask is not None:
                    grad = grad * mask
            masked.append(grad)

        return weights @ torch.stack(masked)

    # ==================== Monitoring ====================

    def _track_correct_probability(self):
        """Average p(correct class) on forget data — should decrease over training."""
        total_correct_prob = 0
        total_samples = 0

        self.module.model.eval()
        with torch.no_grad():
            for client in self.client_list:
                if not getattr(client, "unlearn_flag", False):
                    continue
                for batch_x, batch_y in client.local_training_data:
                    batch_x = fp.Module.change_data_device(batch_x, self.device)
                    batch_y = fp.Module.change_data_device(batch_y, self.device)
                    out = self.module.model(batch_x)
                    probs = F.softmax(out, dim=-1)
                    p_correct = probs[
                        torch.arange(len(batch_y), device=self.device), batch_y
                    ]
                    total_correct_prob += p_correct.sum().item()
                    total_samples += len(batch_y)
        self.module.model.train()

        if total_samples > 0:
            avg = total_correct_prob / total_samples
            self.correct_prob_history.append(avg)
            return avg
        return 0.0

    # ==================== Training Loop ====================

    def _update_loss_alpha(self):
        """Push current alpha(t) to all unlearn clients' loss functions."""
        alpha = self._get_current_alpha()
        for client in self.client_list:
            if getattr(client, "unlearn_flag", False):
                if hasattr(client.criterion, "alpha"):
                    client.criterion.alpha = alpha
        return alpha

    def train_a_round(self):
        """Execute one round of adversarial unlearning."""
        # Anneal loss alpha for this round
        current_alpha = self._update_loss_alpha()

        com_time_start = time.time()
        m_locals, l_locals, g_locals = self.train()
        com_time_end = time.time()

        cal_time_start = time.time()

        if self.model_params is None:
            self.model_params = self.module.span_model_params_to_vec()

        # Refresh masks if needed (uses current dynamic rho)
        self._maybe_refresh_masks()

        # Aggregate with masking
        d = self._aggregate_masked_gradients(g_locals)
        self.update_module(self.module, self.optimizer, self.lr, d)
        self.model_params = self.module.span_model_params_to_vec()

        batch_num = torch.mean(
            torch.Tensor(self.get_clinet_attr("training_batch_num"))
        ).item()
        self.current_training_num += self.epochs * batch_num

        cal_time_end = time.time()
        self.communication_time += com_time_end - com_time_start
        self.computation_time += cal_time_end - cal_time_start

        # Periodic monitoring
        if self.current_comm_round % 10 == 0:
            avg_p = self._track_correct_probability()
            rho = self._get_current_rho()
            print(
                f"Round {self.current_comm_round}: "
                f"p_correct={avg_p:.4f}  rho={rho:.3f}  alpha={current_alpha:.3f}"
            )

    def run(self):
        """Run the adversarial unlearning algorithm."""
        # Create and assign adversarial forgetting loss
        loss = AdversarialForgettingLoss(
            temperature=self.adv_temperature,
            weight=self.adv_weight,
            alpha=self.alpha_init,
        )

        for client in self.client_list:
            if getattr(client, "unlearn_flag", False):
                client.criterion = loss
                print(f"Client {client.id}: AdversarialForgettingLoss assigned")

        # Build initial masks
        self._build_or_refresh_masks()

        print(f"\n{'=' * 60}")
        print("FedGMM-Adversarial")
        print(f"  rho schedule: {self.rho_min:.3f} -> {self.rho_max:.3f} (cosine)")
        print(f"  alpha schedule: {self.alpha_init:.3f} -> {self.alpha_max:.3f} (cosine)")
        print(f"  Temperature: {self.adv_temperature}")
        print(f"  Loss weight: {self.adv_weight}")
        print(f"  Grad divergence: {self.use_grad_divergence}")
        print(f"  Mask EMA beta: {self.mask_ema_beta}")
        print(f"  Mask refresh: every {self.mask_refresh} rounds")
        print(f"{'=' * 60}\n")

        # Main unlearning loop
        while not self.terminated():
            self.train_a_round()

        # Final statistics
        if self.correct_prob_history:
            print(
                f"\np_correct trajectory: "
                f"{self.correct_prob_history[0]:.4f} -> {self.correct_prob_history[-1]:.4f}"
            )
