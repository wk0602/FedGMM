import tfedplat as fp
import time
import torch
import torch.nn.functional as F


class AdversarialForgettingLoss:
    """
    Adversarial Forgetting Loss: Forces model to make confident but WRONG predictions.

    Unlike KL-to-uniform loss which creates uncertainty, this loss actively pushes
    the model to misclassify forget data with high confidence.

    Mathematical formulation:
        L = -log(1 - p_y + ε) + λ_entropy * H(p)

    Where:
        - p_y: probability assigned to correct class
        - H(p): entropy of the prediction distribution
        - λ_entropy: weight for entropy regularization
        - ε: small constant for numerical stability

    Properties:
        1. When p_y → 1 (correct confident): Loss → +∞ (heavily penalized)
        2. When p_y → 0 (wrong confident): Loss → 0 (rewarded)
        3. Entropy term prevents collapse to single wrong class

    Comparison with KL-uniform:
        - KL-uniform: Makes model uncertain (p → 1/K for all classes)
        - Adversarial: Makes model confidently wrong (p_y → 0, p_other → high)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        weight: float = 1.0,
        entropy_weight: float = 0.1,
        margin: float = 0.5,
        eps: float = 1e-8,
        mode: str = "confident_wrong",
    ):
        """
        Initialize Adversarial Forgetting Loss.

        Args:
            temperature: Temperature for softmax (higher = softer predictions)
            weight: Overall loss weight
            entropy_weight: Weight for entropy regularization term
            margin: Margin for confident wrong prediction (p_correct should be < margin)
            eps: Small constant for numerical stability
            mode: Loss mode, one of:
                - "confident_wrong": Push to confident wrong predictions
                - "margin_based": Use margin-based loss for more stable training
                - "hybrid": Combine confident wrong with mild KL-uniform
        """
        self.temperature = temperature
        self.weight = weight
        self.entropy_weight = entropy_weight
        self.margin = margin
        self.eps = eps
        self.mode = mode

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute adversarial forgetting loss.

        Args:
            pred: Model predictions (logits), shape [batch_size, num_classes]
            target: Ground truth labels, shape [batch_size]

        Returns:
            Scalar loss value
        """
        if self.mode == "confident_wrong":
            return self._confident_wrong_loss(pred, target)
        elif self.mode == "margin_based":
            return self._margin_based_loss(pred, target)
        elif self.mode == "hybrid":
            return self._hybrid_loss(pred, target)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _confident_wrong_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Core confident wrong prediction loss.

        Forces the model to assign LOW probability to the correct class
        while maintaining reasonable entropy to avoid collapse.
        """
        # Apply temperature scaling
        scaled_logits = pred / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)

        # Get probability of correct class
        batch_size = pred.size(0)
        correct_probs = probs[torch.arange(batch_size, device=pred.device), target]

        # Main loss: minimize probability of correct class
        # -log(1 - p_correct) → 0 when p_correct → 0 (confident wrong)
        # -log(1 - p_correct) → ∞ when p_correct → 1 (confident correct)
        adversarial_loss = -torch.log(1.0 - correct_probs + self.eps)

        # Entropy regularization: encourage diverse wrong predictions
        # This prevents the model from always predicting a single wrong class
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)

        # Combine: adversarial loss - entropy (we want to maximize entropy)
        # Subtracting entropy encourages higher entropy (more diverse predictions)
        total_loss = adversarial_loss - self.entropy_weight * entropy

        return self.weight * total_loss.mean()

    def _margin_based_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Margin-based adversarial loss for more stable training.

        Uses a margin to ensure the correct class probability stays below threshold.
        """
        scaled_logits = pred / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)

        batch_size = pred.size(0)
        correct_probs = probs[torch.arange(batch_size, device=pred.device), target]

        # Margin loss: penalize when p_correct > margin
        # Only apply loss when the model is too confident about correct class
        margin_violation = F.relu(correct_probs - self.margin)
        adversarial_loss = margin_violation**2  # Squared for smoothness

        # Entropy regularization
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)

        total_loss = adversarial_loss - self.entropy_weight * entropy

        return self.weight * total_loss.mean()

    def _hybrid_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Hybrid loss combining confident wrong with mild KL-uniform.

        This provides a balance between aggressive forgetting and stability.
        """
        scaled_logits = pred / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs = F.log_softmax(scaled_logits, dim=-1)

        batch_size = pred.size(0)
        num_classes = pred.size(1)

        # Confident wrong component
        correct_probs = probs[torch.arange(batch_size, device=pred.device), target]
        confident_wrong_loss = -torch.log(1.0 - correct_probs + self.eps)

        # KL-uniform component (mild uncertainty)
        uniform = torch.full_like(probs, 1.0 / num_classes)
        kl_uniform_loss = F.kl_div(log_probs, uniform, reduction="none").sum(dim=-1)

        # Combine with more weight on confident wrong
        total_loss = 0.7 * confident_wrong_loss + 0.3 * kl_uniform_loss

        return self.weight * total_loss.mean()


class FedGMM_Adversarial(fp.UnlearnAlgorithm):
    """
    Federated Gradient Masking & Modification with Adversarial Forgetting Loss.

    This algorithm improves upon FedGMM by replacing the KL-to-uniform loss with
    an Adversarial Forgetting Loss that forces confident wrong predictions.

    Key improvements over original FedGMM:
        1. Stronger forgetting: Confident wrong > Uncertain predictions
        2. More thorough backdoor removal: Actively breaks pattern associations
        3. Adaptive masking: Can optionally use gradient divergence for mask selection

    Algorithm Overview:
        1. Identify sensitive parameters using gradient magnitude (same as FedGMM)
        2. Apply Adversarial Forgetting Loss on forget data
        3. Aggregate with gradient masking
        4. Optional: Use gradient divergence for smarter parameter selection
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
            name,
            data_loader,
            module,
            device,
            train_setting,
            client_num,
            client_list,
            online_client_num,
            save_model,
            max_comm_round,
            max_training_num,
            epochs,
            save_name,
            outFunc,
            write_log,
            dishonest,
            test_conflicts,
            params,
        )

        # Gradient masking parameters (inherited from FedGMM concept)
        self.mask_dict = {}  # client_id -> mask vector
        self.rho = float(params.get("adv_rho", 0.1)) if params is not None else 0.1
        self.mask_refresh = (
            int(params.get("adv_mask_refresh", 5)) if params is not None else 5
        )

        # Adversarial forgetting loss parameters
        self.adv_weight = (
            float(params.get("adv_weight", 1.0)) if params is not None else 1.0
        )
        self.adv_temperature = (
            float(params.get("adv_temperature", 2.0)) if params is not None else 2.0
        )
        self.adv_entropy_weight = (
            float(params.get("adv_entropy_weight", 0.1)) if params is not None else 0.1
        )
        self.adv_margin = (
            float(params.get("adv_margin", 0.3)) if params is not None else 0.3
        )
        self.adv_mode = (
            str(params.get("adv_mode", "confident_wrong"))
            if params is not None
            else "confident_wrong"
        )

        # Gradient divergence masking (optional enhancement)
        self.use_grad_divergence = (
            bool(params.get("adv_use_grad_divergence", False))
            if params is not None
            else False
        )

        self.model_params = self.module.span_model_params_to_vec()

        # Statistics tracking
        self.correct_prob_history = []  # Track average correct class probability

        # EMA mask stabilization:
        # Maintain a smoothed importance score per unlearn client to avoid mask "thrashing"
        # due to noisy per-round gradients.
        self.mask_score_ema = {}  # client_id -> score vector (same dim as grad vec)
        self.mask_ema_beta = (
            float(params.get("adv_mask_ema_beta", 0.0)) if params is not None else 0.0
        )
        # Backward-compatible: beta <= 0 disables EMA and uses instantaneous score.
        # Recommended range: 0.9 ~ 0.99.

    def _create_adversarial_loss(self) -> AdversarialForgettingLoss:
        """Create adversarial forgetting loss with current parameters."""
        return AdversarialForgettingLoss(
            temperature=self.adv_temperature,
            weight=self.adv_weight,
            entropy_weight=self.adv_entropy_weight,
            margin=self.adv_margin,
            mode=self.adv_mode,
        )

    def _compute_client_gradient(self, client):
        """
        Compute gradient for a client on their local data.

        This is used for gradient-based parameter selection.
        """
        msg = {
            "command": "cal_gradient_loss",
            "epochs": self.epochs,
            "lr": self.lr,
            "target_module": self.module,
        }
        client.get_message(msg)
        msg = {"command": "require_gradient_loss"}
        result = client.get_message(msg)
        return result["g_local"]

    def _compute_gradient_divergence_mask(self, unlearn_client):
        """
        Compute mask based on gradient divergence between forget and retain data.

        This method identifies parameters that are:
        - Important for forget data (high gradient on forget data)
        - Less important for retain data (different gradient direction)

        Such parameters are likely backdoor-specific and should be prioritized for unlearning.
        """
        # Get gradient on forget data
        forget_grad = self._compute_client_gradient(unlearn_client)

        # Get average gradient on retain data from other clients
        retain_grads = []
        for client in self.client_list:
            if not getattr(client, "unlearn_flag", False):
                retain_grads.append(self._compute_client_gradient(client))

        if len(retain_grads) == 0:
            # If no retain clients, fall back to gradient magnitude
            return self._compute_gradient_magnitude_mask(forget_grad)

        retain_grad_avg = torch.stack(retain_grads).mean(dim=0)

        # Compute divergence score
        # High divergence = gradients point in different directions
        # This indicates parameters important for backdoor but not for normal task
        divergence = -forget_grad * retain_grad_avg  # Positive when opposite signs
        divergence = divergence + torch.abs(forget_grad)  # Also consider magnitude

        # Normalize
        divergence = (divergence - divergence.min()) / (
            divergence.max() - divergence.min() + 1e-8
        )

        # Select top-k most divergent parameters
        k = max(1, int(len(divergence) * self.rho))
        _, indices = torch.topk(divergence, k=k, largest=True)

        mask = torch.zeros_like(divergence, device=self.device)
        mask[indices] = 1.0

        return mask

    def _compute_gradient_divergence_score(self, unlearn_client) -> torch.Tensor:
        """
        Compute a continuous divergence score for each parameter (higher = more forget-specific).

        This is the same idea as _compute_gradient_divergence_mask(), but returns a score vector
        so we can apply EMA smoothing before selecting top-k.
        """
        forget_grad = self._compute_client_gradient(unlearn_client)

        retain_grads = []
        for client in self.client_list:
            if not getattr(client, "unlearn_flag", False):
                retain_grads.append(self._compute_client_gradient(client))

        if len(retain_grads) == 0:
            score = torch.abs(forget_grad)
        else:
            retain_grad_avg = torch.stack(retain_grads).mean(dim=0)
            score = -forget_grad * retain_grad_avg
            score = score + torch.abs(forget_grad)

        # Normalize to [0, 1] for stability
        score = (score - score.min()) / (score.max() - score.min() + 1e-8)
        return score

    def _compute_gradient_magnitude_score(self, grad_vec: torch.Tensor) -> torch.Tensor:
        """Continuous magnitude score for each parameter."""
        score = torch.abs(grad_vec)
        score = (score - score.min()) / (score.max() - score.min() + 1e-8)
        return score

    def _score_to_mask(self, score: torch.Tensor) -> torch.Tensor:
        """Convert a continuous score vector into a binary top-rho mask."""
        k = max(1, int(len(score) * self.rho))
        _, indices = torch.topk(score, k=k, largest=True)
        mask = torch.zeros_like(score, device=self.device)
        mask[indices] = 1.0
        return mask

    def _update_ema_score(self, client_id: int, score: torch.Tensor) -> torch.Tensor:
        """
        Update and return the (optionally) EMA-smoothed score for a client.

        If self.mask_ema_beta <= 0, EMA is disabled and 'score' is returned directly.
        """
        beta = float(self.mask_ema_beta)
        if beta <= 0.0:
            return score

        prev = self.mask_score_ema.get(client_id)
        if prev is None:
            self.mask_score_ema[client_id] = score.detach()
        else:
            self.mask_score_ema[client_id] = beta * prev + (1.0 - beta) * score.detach()
        return self.mask_score_ema[client_id]

    def _compute_gradient_magnitude_mask(self, grad_vec):
        """
        Compute mask based on gradient magnitude (standard approach).
        """
        k = max(1, int(len(grad_vec) * self.rho))
        _, indices = torch.topk(torch.abs(grad_vec), k=k, largest=True, sorted=False)

        mask = torch.zeros_like(grad_vec, device=self.device)
        mask[indices] = 1.0

        return mask

    def _build_or_refresh_masks(self):
        """
        Generate masks for unlearn clients.

        Uses either gradient divergence (if enabled) or gradient magnitude.
        """
        for client in self.client_list:
            if not getattr(client, "unlearn_flag", False):
                continue

            if self.use_grad_divergence:
                score = self._compute_gradient_divergence_score(client)
            else:
                grad_vec = self._compute_client_gradient(client)
                score = self._compute_gradient_magnitude_score(grad_vec)

            score_smoothed = self._update_ema_score(client.id, score)
            self.mask_dict[client.id] = self._score_to_mask(score_smoothed)

    def _maybe_refresh_masks(self):
        """Refresh masks periodically or when needed."""
        if not self.mask_dict:
            self._build_or_refresh_masks()
            return
        if self.mask_refresh > 0 and self.current_comm_round > 0:
            if self.current_comm_round % self.mask_refresh == 0:
                self._build_or_refresh_masks()

    def _aggregate_masked_gradients(self, g_locals):
        """
        Apply masks on unlearn clients then weighted average.

        For unlearn clients: gradient is masked to focus on sensitive parameters
        For retain clients: gradient is used as-is
        """
        weights = (
            torch.Tensor(self.get_clinet_attr("local_training_number"))
            .float()
            .to(self.device)
        )
        weights = weights / torch.sum(weights)

        masked = []
        for idx, client in enumerate(self.online_client_list):
            grad = g_locals[idx]
            if getattr(client, "unlearn_flag", False):
                mask = self.mask_dict.get(client.id)
                if mask is not None:
                    grad = grad * mask
            masked.append(grad)

        masked = torch.stack(masked)
        return weights @ masked

    def _track_correct_probability(self):
        """
        Track average probability assigned to correct class for monitoring.

        This helps verify that the adversarial loss is working
        (correct probability should decrease over time).
        """
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
                    correct_probs = probs[
                        torch.arange(len(batch_y), device=self.device), batch_y
                    ]

                    total_correct_prob += correct_probs.sum().item()
                    total_samples += len(batch_y)

        self.module.model.train()

        if total_samples > 0:
            avg_correct_prob = total_correct_prob / total_samples
            self.correct_prob_history.append(avg_correct_prob)
            return avg_correct_prob
        return 0.0

    def train_a_round(self):
        """Execute one round of adversarial unlearning."""
        com_time_start = time.time()
        m_locals, l_locals, g_locals = self.train()
        com_time_end = time.time()

        cal_time_start = time.time()

        # Ensure model_params is set
        if self.model_params is None:
            self.model_params = self.module.span_model_params_to_vec()

        # Refresh masks if needed
        self._maybe_refresh_masks()

        # Aggregate with masking
        d = self._aggregate_masked_gradients(g_locals)
        lr = self.lr
        self.update_module(self.module, self.optimizer, lr, d)
        self.model_params = self.module.span_model_params_to_vec()

        batch_num = torch.mean(
            torch.Tensor(self.get_clinet_attr("training_batch_num"))
        ).item()
        self.current_training_num += self.epochs * batch_num

        cal_time_end = time.time()
        self.communication_time += com_time_end - com_time_start
        self.computation_time += cal_time_end - cal_time_start

        # Track correct probability for monitoring (every 10 rounds)
        if self.current_comm_round % 10 == 0:
            avg_correct_prob = self._track_correct_probability()
            print(
                f"Round {self.current_comm_round}: Avg correct class prob = {avg_correct_prob:.4f}"
            )

    def run(self):
        """Run the adversarial unlearning algorithm."""
        # Assign adversarial forgetting loss to unlearn clients
        adversarial_loss = self._create_adversarial_loss()

        for client in self.client_list:
            if getattr(client, "unlearn_flag", False):
                client.criterion = adversarial_loss
                print(
                    f"Client {client.id}: Assigned AdversarialForgettingLoss (mode={self.adv_mode})"
                )

        # Build initial masks
        self._build_or_refresh_masks()

        print(f"\n{'=' * 60}")
        print("FedGMM-Adversarial Starting")
        print(f"  Mode: {self.adv_mode}")
        print(f"  Temperature: {self.adv_temperature}")
        print(f"  Entropy Weight: {self.adv_entropy_weight}")
        print(f"  Mask Ratio (rho): {self.rho}")
        print(f"  Use Gradient Divergence: {self.use_grad_divergence}")
        print(f"{'=' * 60}\n")

        # Main unlearning loop
        while not self.terminated():
            self.train_a_round()

        # Print final statistics
        if self.correct_prob_history:
            print(
                f"\nCorrect probability trajectory: {self.correct_prob_history[:5]} ... {self.correct_prob_history[-5:]}"
            )
            print(
                f"Initial: {self.correct_prob_history[0]:.4f} → Final: {self.correct_prob_history[-1]:.4f}"
            )
