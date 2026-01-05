import tfedplat as fp
import time
import torch
import torch.nn.functional as F


class FedGMM(fp.UnlearnAlgorithm):
    """
    Federated Gradient Masking & Modification (Fed-GMM)
    - Unlearning stage: identify sensitive parameters via gradient magnitude,
      mask only top-rho weights for updates, use KL-to-uniform loss to remove
      information about the forget set while keeping anchors untouched.
    """

    def __init__(
        self,
        name="FedGMM",
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
        self.mask_dict = {}  # client_id -> mask vector
        self.rho = float(params.get("gmm_rho", 0.05)) if params is not None else 0.05
        self.mask_refresh = (
            int(params.get("gmm_mask_refresh", 0)) if params is not None else 0
        )
        self.kl_weight = (
            float(params.get("gmm_kl_weight", 1.0)) if params is not None else 1.0
        )
        self.temperature = (
            float(params.get("gmm_temperature", 1.0)) if params is not None else 1.0
        )
        self.model_params = self.module.span_model_params_to_vec()

    class UniformKLLoss:
        def __init__(
            self, reduction="mean", temperature=1.0, weight=1.0, ignore_index=-100
        ):
            self.reduction = reduction
            self.temperature = temperature
            self.weight = weight
            self.ignore_index = ignore_index

        def __call__(self, pred, target):
            class_num = int(pred.shape[1])
            log_probs = F.log_softmax(pred / self.temperature, dim=-1)
            uniform = torch.full_like(log_probs, 1.0 / class_num)

            loss_vec = -torch.sum(
                uniform * log_probs, dim=1
            )  # CE to uniform == KL(uniform || pred)
            if self.reduction == "mean":
                return self.weight * torch.mean(loss_vec)
            if self.reduction == "sum":
                return self.weight * torch.sum(loss_vec)
            return self.weight * loss_vec

    def _compute_client_gradient(self, client):
        """
        Run a forward/backward pass on client's local data to obtain averaged gradient vector.
        Reuses existing client routines to stay consistent with clipping and data handling.
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

    def _build_or_refresh_masks(self):
        """Generate masks for unlearn clients based on gradient magnitude (top-rho)."""
        for client in self.client_list:
            if not getattr(client, "unlearn_flag", False):
                continue
            grad_vec = self._compute_client_gradient(client)
            k = max(1, int(len(grad_vec) * self.rho))
            values, indices = torch.topk(
                torch.abs(grad_vec), k=k, largest=True, sorted=False
            )
            mask = torch.zeros_like(grad_vec, device=self.device)
            mask[indices] = 1.0
            self.mask_dict[client.id] = mask

    def _maybe_refresh_masks(self):
        if not self.mask_dict:
            self._build_or_refresh_masks()
            return
        if self.mask_refresh > 0 and self.current_comm_round > 0:
            if self.current_comm_round % self.mask_refresh == 0:
                self._build_or_refresh_masks()

    def _aggregate_masked_gradients(self, g_locals):
        """Apply masks on unlearn clients then weighted average by local data size."""
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

    def train_a_round(self):
        com_time_start = time.time()
        m_locals, l_locals, g_locals = self.train()
        com_time_end = time.time()

        cal_time_start = time.time()

        # Ensure model_params is set before grad calculations that depend on it.
        if self.model_params is None:
            self.model_params = self.module.span_model_params_to_vec()

        # Unlearning stage: use masked gradients
        self._maybe_refresh_masks()
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

    def run(self):
        # Assign KL-to-uniform loss to unlearn clients
        for client in self.client_list:
            if getattr(client, "unlearn_flag", False):
                client.criterion = self.UniformKLLoss(
                    temperature=self.temperature, weight=self.kl_weight
                )

        # Initial mask build before unlearning rounds
        self._build_or_refresh_masks()

        while not self.terminated():
            self.train_a_round()
