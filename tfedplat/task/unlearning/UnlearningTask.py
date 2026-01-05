import tfedplat as fp
import numpy as np
import torch
import os
import copy
from torchvision.transforms import transforms

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class UnlearningTask(fp.BasicTask):
    def __init__(self, name="UnlearningTask"):
        super().__init__(name)

        self.algorithm.save_folder = (
            self.name
            + "/"
            + self.params["module"]
            + "/"
            + self.data_loader.nickname
            + "/UN"
            + str(self.params["unlearn_cn"])
            + "/E"
            + str(self.params["E"])
            + "/C"
            + str(self.params["C"])
            + "/"
            + self.params["algorithm"]
            + "/"
        )
        unlearn_pretrain_flag = self.params["unlearn_pretrain"]

        pretrained_model_folder = (
            self.name
            + "/"
            + self.params["module"]
            + "/"
            + self.data_loader.nickname
            + "/UN"
            + str(self.params["unlearn_cn"])
            + "/E"
            + str(self.params["E"])
            + "/"
        )
        self.algorithm.pretrained_model_folder = pretrained_model_folder
        if not unlearn_pretrain_flag:
            model_path = (
                pretrained_model_folder
                + f"seed{self.params['seed']}_unlearn_task_pretrained_model.pth"
            )
            if not os.path.exists(pretrained_model_folder):
                os.makedirs(pretrained_model_folder)
            if not os.path.isfile(model_path):
                raise RuntimeError(
                    f"Please put the pretrained model in the path {model_path}."
                )
            self.algorithm.module.model.load_state_dict(torch.load(model_path))
            self.algorithm.init_model_params = (
                self.algorithm.module.span_model_params_to_vec()
            )
            self.algorithm.model_params = (
                self.algorithm.module.span_model_params_to_vec()
            )
        else:
            self.algorithm.save_model = True
            self.algorithm.model_save_name = (
                pretrained_model_folder
                + f"seed{self.params['seed']}_unlearn_task_pretrained_model.pth"
            )

            if isinstance(self.algorithm, fp.UnlearnAlgorithm):
                raise RuntimeError(
                    "When setting unlearn_pretrain_flag=True, you cannot run unlearning FL algorithm."
                )

            self.algorithm.terminate_extra_execute = self.terminate_extra_execute

        self.params["unlearn_client_id_list"] = np.random.choice(
            self.algorithm.client_num, self.params["unlearn_cn"], replace=False
        ).tolist()
        print("Unlearn clients:", self.params["unlearn_client_id_list"])
        self.algorithm.out_log += (
            f"Unlearn clients: {self.params['unlearn_client_id_list']}"
        )

        for client in self.algorithm.client_list:
            if client.id in self.params["unlearn_client_id_list"]:
                setattr(client, "unlearn_flag", True)
                pretrain_attack_portion = 0.8 if unlearn_pretrain_flag else 1.0
                self.modify_client(client, pretrain_attack_portion)
            else:
                setattr(client, "unlearn_flag", False)

    @staticmethod
    def terminate_extra_execute(alg):
        alg.__class__.__bases__[0].terminate_extra_execute(alg)

    def modify_client(self, client, attack_portion):
        setattr(
            client,
            "local_backdoor_test_data",
            copy.deepcopy(client.local_training_data),
        )
        setattr(client, "local_backdoor_test_number", client.local_training_number)

        backdoor = fp.FigRandBackdoor(
            dataloader=self.algorithm.data_loader,
            save_folder=self.algorithm.pretrained_model_folder + "backdoors/",
            save_name=f"client_{client.id}_backdoor",
        )
        backdoor.add_backdoor(client.local_training_data, attack_portion=attack_portion)
        backdoor.add_backdoor(
            client.local_backdoor_test_data, attack_portion=attack_portion
        )
        setattr(client, "backdoor_setting", backdoor)

        client.local_test_data = copy.deepcopy(client.local_training_data)
        client.local_test_number = client.local_training_number

        client.test = self.ClientTest(
            self.algorithm.train_setting, self.algorithm.device
        )

    @staticmethod
    class ClientTest:
        def __init__(self, train_setting, device):
            self.train_setting = train_setting
            self.device = device

            self.metric_history = {
                "training_loss": [],
                "test_loss": [],
                "local_test_number": 0,
                "test_accuracy": [],
                "backdoor_test_loss": [],
                "backdoor_test_accuracy": [],
            }

        def run(self, client):
            client.test_module.model.eval()
            criterion = self.train_setting["criterion"].to(self.device)

            self.metric_history["training_loss"].append(
                float(client.upload_loss) if client.upload_loss is not None else None
            )

            metric_dict = {"test_loss": 0, "correct": 0}

            correct_metric = fp.Correct()

            with torch.no_grad():
                self.metric_history["local_test_number"] = client.local_test_number
                for [batch_x, batch_y] in client.local_test_data:
                    batch_x = fp.Module.change_data_device(batch_x, self.device)
                    batch_y = fp.Module.change_data_device(batch_y, self.device)

                    out = client.test_module.model(batch_x)
                    loss = criterion(out, batch_y).item()
                    metric_dict["test_loss"] += float(loss) * batch_y.shape[0]
                    metric_dict["correct"] += correct_metric.calc(out, batch_y)

                self.metric_history["test_loss"].append(
                    round(metric_dict["test_loss"] / client.local_test_number, 4)
                )
                self.metric_history["test_accuracy"].append(
                    100 * metric_dict["correct"] / client.local_test_number
                )

                metric_dict = {"test_loss": 0, "correct": 0}
                for [batch_x, batch_y] in client.local_backdoor_test_data:
                    batch_x = fp.Module.change_data_device(batch_x, self.device)
                    batch_y = fp.Module.change_data_device(batch_y, self.device)

                    out = client.test_module.model(batch_x)
                    loss = criterion(out, batch_y).item()
                    metric_dict["test_loss"] += float(loss) * batch_y.shape[0]
                    metric_dict["correct"] += correct_metric.calc(out, batch_y)

                self.metric_history["backdoor_test_loss"].append(
                    round(
                        metric_dict["test_loss"] / client.local_backdoor_test_number, 4
                    )
                )
                self.metric_history["backdoor_test_accuracy"].append(
                    100 * metric_dict["correct"] / client.local_backdoor_test_number
                )

    @staticmethod
    def outFunc(alg):
        unlearned_client_loss_list = []
        retained_client_loss_list = []
        for i, metric_history in enumerate(alg.metric_log["client_metric_history"]):
            training_loss = metric_history["training_loss"][-1]
            if training_loss is None:
                continue
            if alg.client_list[i].unlearn_flag:
                unlearned_client_loss_list.append(training_loss)
            else:
                retained_client_loss_list.append(training_loss)

        unlearned_client_local_acc_list = []
        retained_client_local_acc_list = []
        for i, metric_history in enumerate(alg.metric_log["client_metric_history"]):
            test_acc = metric_history["test_accuracy"][-1]
            if alg.client_list[i].unlearn_flag:
                unlearned_client_local_acc_list.append(test_acc)
            else:
                retained_client_local_acc_list.append(test_acc)

        unlearned_client_local_backdoor_acc_list = []
        for i, metric_history in enumerate(alg.metric_log["client_metric_history"]):
            if alg.client_list[i].unlearn_flag:
                test_acc = metric_history["backdoor_test_accuracy"][-1]
                unlearned_client_local_backdoor_acc_list.append(test_acc)
        unlearned_client_local_acc_list = np.array(unlearned_client_local_acc_list)
        retained_client_local_acc_list = np.array(retained_client_local_acc_list)
        unlearned_client_local_backdoor_acc_list = np.array(
            unlearned_client_local_backdoor_acc_list
        )

        def cal_fairness(values):
            p = np.ones(len(values))
            fairness = np.arccos(
                values @ p / (np.linalg.norm(values) * np.linalg.norm(p))
            )
            return fairness

        unlearned_client_fairness = cal_fairness(unlearned_client_local_acc_list)
        retained_client_fairness = cal_fairness(retained_client_local_acc_list)

        out_log = ""
        out_log += alg.save_name + " " + alg.data_loader.nickname + "\n"
        out_log += "Lr: " + str(alg.lr) + "\n"
        out_log += (
            "round {}".format(alg.current_comm_round)
            + " training_num {}".format(alg.current_training_num)
            + "\n"
        )
        out_log += (
            f"Unlearned Client Mean Global Test loss: {format(np.mean(unlearned_client_loss_list), '.6f')}"
            + "\n"
            if len(unlearned_client_loss_list) > 0
            else ""
        )
        out_log += (
            f"Unlearned Client Local Test Acc: {format(np.mean(unlearned_client_local_acc_list / 100), '.3f')}({format(np.std(unlearned_client_local_acc_list / 100), '.3f')}), angle: {format(unlearned_client_fairness, '.6f')}, min: {format(np.min(unlearned_client_local_acc_list), '.6f')}, max: {format(np.max(unlearned_client_local_acc_list), '.6f')}"
            + "\n"
            if len(unlearned_client_local_acc_list) > 0
            else ""
        )
        out_log += (
            f"ASR: {format(np.mean(unlearned_client_local_backdoor_acc_list / 100), '.3f')}({format(np.std(unlearned_client_local_backdoor_acc_list / 100), '.3f')}), min: {format(np.min(unlearned_client_local_backdoor_acc_list), '.6f')}, max: {format(np.max(unlearned_client_local_backdoor_acc_list), '.6f')}"
            + "\n"
            if len(unlearned_client_local_backdoor_acc_list) > 0
            else ""
        )
        out_log += (
            f"Retained Client Mean Global Test loss: {format(np.mean(retained_client_loss_list), '.6f')}"
            + "\n"
            if len(retained_client_loss_list) > 0
            else ""
        )
        out_log += (
            f"Retained Client Local Test Acc: {format(np.mean(retained_client_local_acc_list / 100), '.3f')}({format(np.std(retained_client_local_acc_list / 100), '.3f')}), angle: {format(retained_client_fairness, '.6f')}, min: {format(np.min(retained_client_local_acc_list), '.6f')}, max: {format(np.max(retained_client_local_acc_list), '.6f')}"
            + "\n"
        )
        out_log += f"communication_time: {alg.communication_time}, computation_time: {alg.computation_time} \n"
        out_log += "\n"
        alg.out_log = out_log + alg.out_log
        print(str(alg.name))
        print(out_log)

    def read_params(self, return_parser=False):
        parser = super().read_params(return_parser=True)

        parser.add_argument(
            "--unlearn_cn", help="unlearn client num", type=int, default=None
        )
        parser.add_argument(
            "--unlearn_pretrain",
            help="pretrain the model before unlearning",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--UR",
            help="Unlearning round, must be smaller than R",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--r_lr",
            help="Learning rate in the post-training (deprecated, no recovery stage)",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--gmm_rho", help="Top-rho mask ratio for FedGMM", type=float, default=None
        )
        parser.add_argument(
            "--gmm_mask_refresh",
            help="Mask refresh interval (0 means no refresh)",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--gmm_kl_weight",
            help="KL-to-uniform loss weight for FedGMM",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--gmm_temperature",
            help="Temperature for KL-to-uniform",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--early_stop", help="Enable early stopping", type=bool, default=None
        )
        parser.add_argument(
            "--early_stop_threshold",
            help="ASR threshold for early stopping (stop when ASR <= threshold)",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--early_stop_patience",
            help="Patience for early stopping (stop if no improvement for N rounds)",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--early_stop_min_delta",
            help="Minimum ASR change to count as improvement",
            type=float,
            default=None,
        )

        # FedGMM-Adversarial specific parameters
        parser.add_argument(
            "--adv_rho",
            help="Top-rho mask ratio for FedGMM-Adversarial",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--adv_mask_refresh",
            help="Mask refresh interval for FedGMM-Adversarial",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--adv_weight",
            help="Adversarial forgetting loss weight",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--adv_temperature",
            help="Temperature for adversarial loss softmax",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--adv_entropy_weight",
            help="Entropy regularization weight for adversarial loss",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--adv_margin",
            help="Margin for margin-based adversarial loss mode",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--adv_mode",
            help="Adversarial loss mode: confident_wrong, margin_based, or hybrid",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--adv_use_grad_divergence",
            help="Use gradient divergence for mask selection",
            type=str,
            default=None,
        )

        try:
            if return_parser:
                return parser
            else:
                args = parser.parse_args()

                # Default values for unlearning-specific parameters
                unlearning_defaults = {
                    "unlearn_cn": 0,
                    "unlearn_pretrain": False,
                    "UR": 2000,
                    "r_lr": -1,
                    # FedGMM parameters
                    "gmm_rho": 0.05,
                    "gmm_mask_refresh": 0,
                    "gmm_kl_weight": 1.0,
                    "gmm_temperature": 1.0,
                    # FedGMM-Adversarial parameters
                    "adv_rho": 0.15,
                    "adv_mask_refresh": 5,
                    "adv_weight": 1.0,
                    "adv_temperature": 2.0,
                    "adv_entropy_weight": 0.1,
                    "adv_margin": 0.3,
                    "adv_mode": "confident_wrong",
                    "adv_use_grad_divergence": False,
                    # Early stopping
                    "early_stop": True,
                    "early_stop_threshold": 0.05,
                    "early_stop_patience": 10,
                    "early_stop_min_delta": 0.001,
                }

                # Determine config file path and set default if not specified
                import os
                import sys

                config_path = args.config

                # If no config specified, try to find default config_unlearning.yaml
                if not config_path:
                    # Try to find config_unlearning.yaml in script directory or current directory
                    # Get project root (assuming run_UnlearningTask.py is in project root)
                    script_dir = os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        )
                    )
                    possible_paths = [
                        os.path.join(
                            script_dir, "config_unlearning.yaml"
                        ),  # Project root
                        os.path.join(
                            os.getcwd(), "config_unlearning.yaml"
                        ),  # Current working directory
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            config_path = os.path.abspath(path)  # Use absolute path
                            # Add to sys.argv so parent class will see it when parsing
                            # Check if --config is already in sys.argv
                            has_config = False
                            for i, arg in enumerate(sys.argv):
                                if arg == "--config":
                                    has_config = True
                                    # Update existing config path
                                    if i + 1 < len(sys.argv):
                                        sys.argv[i + 1] = config_path
                                    else:
                                        sys.argv.append(config_path)
                                    break

                            if not has_config:
                                # Insert --config and path after script name
                                sys.argv.insert(1, "--config")
                                sys.argv.insert(2, config_path)
                            # Update args.config for consistency
                            args.config = config_path
                            break

                # Get base params (which already handles config file and command line args)
                # Parent class will load the config file and merge all parameters
                # Note: We modify sys.argv above so parent class will see the --config argument
                params = super().read_params(return_parser=False)

                # Add unlearning defaults for any missing parameters
                for key, default_value in unlearning_defaults.items():
                    if key not in params:
                        params[key] = default_value

                # Override with command line arguments (if provided)
                args_dict = vars(args)
                for key in unlearning_defaults.keys():
                    if args_dict.get(key) is not None:
                        params[key] = args_dict[key]

                if params["UR"] > params["R"]:
                    raise RuntimeError("The parameter of UR must not be bigger than R.")

                # Convert early_stop to boolean if it's a string
                if isinstance(params.get("early_stop"), str):
                    params["early_stop"] = params["early_stop"].lower() in (
                        "true",
                        "1",
                        "yes",
                        "on",
                    )

                # Convert unlearn_pretrain to boolean if it's a string
                if isinstance(params.get("unlearn_pretrain"), str):
                    params["unlearn_pretrain"] = params["unlearn_pretrain"].lower() in (
                        "true",
                        "1",
                        "yes",
                        "on",
                    )

                # Convert adv_use_grad_divergence to boolean if it's a string
                if isinstance(params.get("adv_use_grad_divergence"), str):
                    params["adv_use_grad_divergence"] = params[
                        "adv_use_grad_divergence"
                    ].lower() in (
                        "true",
                        "1",
                        "yes",
                        "on",
                    )

                return params
        except IOError as msg:
            parser.error(str(msg))


if __name__ == "__main__":
    my_task = UnlearningTask()
    my_task.run()
