import tfedplat as fp
import numpy as np
import argparse
import torch
import sys
import os
import yaml

torch.multiprocessing.set_sharing_strategy("file_system")


class BasicTask:
    def __init__(self, name="BasicTask"):
        self.name = name
        self.params = self.read_params()
        self.data_loader, self.algorithm = self.initialize(self.params)

        self.algorithm.save_folder = (
            self.name
            + "/"
            + self.params["module"]
            + "/"
            + self.data_loader.nickname
            + "/C"
            + str(self.params["C"])
            + "/"
            + self.params["algorithm"]
            + "/"
        )

        self.algorithm.save_name = (
            "seed"
            + str(self.params["seed"])
            + " N"
            + str(self.data_loader.pool_size)
            + " C"
            + str(self.params["C"])
            + " "
            + self.algorithm.save_name
        )
        if self.params["load_pretrained"]:
            model_path = (
                self.name
                + "/"
                + self.params["module"]
                + "/"
                + self.data_loader.nickname
                + "/pretrained_model.pth"
            )
            if os.path.isfile(model_path):
                print("Find pretrained model")
                self.algorithm.module.model.load_state_dict(torch.load(model_path))
                self.algorithm.module.model.to(self.algorithm.device)

    def run(self):
        self.algorithm.start_running()

    def __str__(self):
        print(self.params)

    @staticmethod
    def outFunc(alg):
        loss_list = []
        for i, metric_history in enumerate(alg.metric_log["client_metric_history"]):
            training_loss = metric_history["training_loss"][-1]
            if training_loss is None:
                continue
            loss_list.append(training_loss)
        loss_list = np.array(loss_list)

        local_acc_list = []
        for i, metric_history in enumerate(alg.metric_log["client_metric_history"]):
            local_acc_list.append(metric_history["test_accuracy"][-1])
        local_acc_list = np.array(local_acc_list)
        p = np.ones(len(local_acc_list))
        local_acc_fairness = np.arccos(
            local_acc_list @ p / (np.linalg.norm(local_acc_list) * np.linalg.norm(p))
        )

        out_log = ""
        out_log += alg.save_name + " " + alg.data_loader.nickname + "\n"
        out_log += "Lr: " + str(alg.lr) + "\n"
        out_log += (
            "round {}".format(alg.current_comm_round)
            + " training_num {}".format(alg.current_training_num)
            + "\n"
        )
        out_log += (
            f"Mean Global Test loss: {format(np.mean(loss_list), '.6f')}" + "\n"
            if len(loss_list) > 0
            else ""
        )
        out_log += "global model test: \n"
        out_log += (
            f"Local Test Acc: {format(np.mean(local_acc_list / 100), '.3f')}({format(np.std(local_acc_list / 100), '.3f')}), angle: {format(local_acc_fairness, '.6f')}, min: {format(np.min(local_acc_list), '.6f')}, max: {format(np.max(local_acc_list), '.6f')}"
            + "\n"
        )
        out_log += f"communication_time: {alg.communication_time}, computation_time: {alg.computation_time} \n"
        out_log += "\n"
        alg.out_log = out_log + alg.out_log
        print(out_log)

    def read_params(self, return_parser=False):
        parser = argparse.ArgumentParser()

        # Add config file argument
        parser.add_argument(
            "--config", help="Path to YAML configuration file", type=str, default=None
        )

        parser.add_argument("--seed", help="seed", type=int, default=None)

        parser.add_argument(
            "--device", help="device: -1, 0, 1, or ...", type=int, default=None
        )

        parser.add_argument("--module", help="module name;", type=str, default=None)

        parser.add_argument(
            "--algorithm", help="algorithm name;", type=str, default=None
        )

        parser.add_argument(
            "--dataloader", help="dataloader name;", type=str, default=None
        )

        parser.add_argument("--SN", help="split num", type=int, default=None)

        parser.add_argument("--PN", help="pick num", type=int, default=None)

        parser.add_argument("--B", help="batch size", type=int, default=None)

        parser.add_argument("--NC", help="client_class_num", type=int, default=None)

        parser.add_argument(
            "--balance",
            help="balance or not for pathological separation",
            type=str,
            default=None,
        )

        parser.add_argument(
            "--Diralpha", help="alpha parameter for dirichlet", type=float, default=None
        )

        parser.add_argument(
            "--types", help="dataloader label types;", type=str, default=None
        )

        parser.add_argument("--N", help="client num", type=int, default=None)

        parser.add_argument(
            "--C", help="select client proportion", type=float, default=None
        )

        parser.add_argument("--R", help="communication round", type=int, default=None)

        parser.add_argument("--E", help="local epochs", type=int, default=None)

        parser.add_argument(
            "--test_interval", help="test interval", type=int, default=None
        )

        parser.add_argument(
            "--test_conflicts", help="test conflicts", type=str, default=None
        )

        parser.add_argument("--step_type", help="step type", type=str, default=None)

        parser.add_argument("--test_module", help="test module", type=str, default=None)

        parser.add_argument("--lr", help="learning rate", type=float, default=None)
        parser.add_argument(
            "--decay", help="learning rate decay", type=float, default=None
        )
        parser.add_argument("--momentum", help="momentum", type=float, default=None)

        parser.add_argument(
            "--theta", help="fairness angle of FedMDFG", type=float, default=None
        )
        parser.add_argument(
            "--s", help="line search parameter of FedMDFG", type=int, default=None
        )

        parser.add_argument(
            "--alpha", help="alpha of FedFV/APFL", type=float, default=None
        )
        parser.add_argument(
            "--tau", help="parameter tau in FedFV/FedRep", type=int, default=None
        )

        parser.add_argument("--beta", help="beta of FedFa", type=float, default=None)
        parser.add_argument(
            "--gamma", help="parameter gamma in FedFa", type=float, default=None
        )

        parser.add_argument(
            "--lam",
            help="parameter tau in Ditto/FedAMP/pFedMe/pFedGF",
            type=float,
            default=None,
        )

        parser.add_argument(
            "--epsilon", help="parameter epsilon in FedMGDA+", type=float, default=None
        )

        parser.add_argument(
            "--q", help="parameter q in qFedAvg", type=float, default=None
        )

        parser.add_argument("--t", help="parameter t in TERM", type=float, default=None)

        parser.add_argument(
            "--mu", help="parameter mu in FedProx", type=float, default=None
        )

        parser.add_argument(
            "--dishonest_num", help="dishonest number", type=int, default=None
        )
        parser.add_argument(
            "--scaled_update", help="scaled update attack", type=str, default=None
        )
        parser.add_argument(
            "--random_update", help="random update attack", type=str, default=None
        )
        parser.add_argument(
            "--zero_update", help="zero update attack", type=str, default=None
        )

        parser.add_argument(
            "--load_pretrained",
            help="Load the pretrained model",
            type=str,
            default=None,
        )

        parser.add_argument(
            "--g_clip",
            help="parameter gradient clipping when using gradient ascent.",
            type=float,
            default=None,
        )

        parser.add_argument("--save_model", help="save model", type=str, default=None)

        try:
            if return_parser:
                return parser
            else:
                args = parser.parse_args()

                # Default values
                defaults = {
                    "seed": 1,
                    "device": 0,
                    "module": "CNN_CIFAR10_FedFV",
                    "algorithm": "FedAvg",
                    "dataloader": "DataLoader_cifar10_non_iid",
                    "SN": 200,
                    "PN": 2,
                    "B": 50,
                    "NC": 2,
                    "balance": "True",
                    "Diralpha": 0.1,
                    "types": "default_type",
                    "N": 100,
                    "C": 1.0,
                    "R": 3000,
                    "E": 1,
                    "test_interval": 1,
                    "test_conflicts": "False",
                    "step_type": "bgd",
                    "test_module": "module",
                    "lr": 0.1,
                    "decay": 0.999,
                    "momentum": 0.0,
                    "theta": 11.25,
                    "s": 1,
                    "alpha": 0.1,
                    "tau": 1,
                    "beta": 0.5,
                    "gamma": 0.9,
                    "lam": 0.1,
                    "epsilon": 0.1,
                    "q": 0.1,
                    "t": 1.0,
                    "mu": 0.0,
                    "dishonest_num": 0,
                    "scaled_update": "None",
                    "random_update": "None",
                    "zero_update": "None",
                    "load_pretrained": False,
                    "g_clip": -1.0,
                    "save_model": "True",
                }

                # Load config file if provided
                config_params = {}
                if args.config:
                    config_path = args.config
                    if not os.path.isabs(config_path):
                        # If relative path, try relative to current working directory
                        config_path = os.path.join(os.getcwd(), config_path)
                    if os.path.exists(config_path):
                        with open(config_path, "r", encoding="utf-8") as f:
                            config_params = yaml.safe_load(f) or {}
                    else:
                        raise FileNotFoundError(
                            f"Configuration file not found: {config_path}"
                        )

                # Merge: defaults -> config file -> command line args
                params = defaults.copy()
                params.update(config_params)

                # Override with command line arguments (if provided)
                args_dict = vars(args)
                for key, value in args_dict.items():
                    if value is not None and key != "config":
                        params[key] = value

                # Convert boolean values to strings for parameters that use eval()
                # This is needed because YAML loads booleans as Python bool, but eval() expects strings
                eval_params = [
                    "balance",
                    "test_conflicts",
                    "scaled_update",
                    "random_update",
                    "zero_update",
                    "save_model",
                    "load_pretrained",
                ]
                for key in eval_params:
                    if key in params and isinstance(params[key], bool):
                        params[key] = str(params[key])

                return params
        except IOError as msg:
            parser.error(str(msg))

    def initialize(self, params):
        fp.setup_seed(seed=params["seed"])
        device = torch.device(
            "cuda:" + str(params["device"])
            if torch.cuda.is_available() and params["device"] != -1
            else "cpu"
        )
        Module = getattr(sys.modules["tfedplat"], params["module"])
        module = Module(device)
        Dataloader = getattr(sys.modules["tfedplat"], params["dataloader"])
        data_loader = Dataloader(
            params=params, input_require_shape=module.input_require_shape
        )
        module.generate_model(
            data_loader.input_data_shape, data_loader.target_class_num
        )
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, module.model.parameters()),
            lr=params["lr"],
            momentum=params["momentum"],
        )
        train_setting = {
            "criterion": torch.nn.CrossEntropyLoss(),
            "optimizer": optimizer,
            "lr_decay": params["decay"],
            "step_type": params["step_type"],
            "g_clip": params["g_clip"],
        }
        test_interval = params["test_interval"]
        save_model = params["save_model"]
        dishonest_num = params["dishonest_num"]
        scaled_update = eval(params["scaled_update"])
        if scaled_update is not None:
            scaled_update = float(scaled_update)
        dishonest = {
            "dishonest_num": dishonest_num,
            "scaled_update": scaled_update,
            "random_update": eval(params["random_update"]),
            "zero_update": eval(params["zero_update"]),
        }
        test_conflicts = eval(params["test_conflicts"])
        Algorithm = getattr(sys.modules["tfedplat"], params["algorithm"])
        algorithm = Algorithm(
            data_loader=data_loader,
            module=module,
            device=device,
            train_setting=train_setting,
            client_num=data_loader.pool_size,
            online_client_num=int(data_loader.pool_size * params["C"]),
            save_model=save_model,
            max_comm_round=params["R"],
            max_training_num=None,
            epochs=params["E"],
            outFunc=self.outFunc,
            write_log=True,
            dishonest=dishonest,
            test_conflicts=test_conflicts,
            params=params,
        )
        algorithm.test_interval = test_interval
        return data_loader, algorithm


if __name__ == "__main__":
    my_task = BasicTask()
    my_task.run()
