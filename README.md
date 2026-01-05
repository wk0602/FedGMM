# Fed-GMM: Federated Gradient Masking & Modification

轻量级联邦遗忘实现：用梯度敏感度生成稀疏掩码，仅更新最敏感的参数，并用 KL-to-uniform 让模型在待遗忘数据上"拉平"输出，减少对其余任务的伤害。

## 快速开始

### 方式一：使用配置文件（推荐）

1) 预训练全局模型（保持原有联邦训练不变）
```bash
python run_UnlearningTask.py --config config_pretrain.yaml
```

或者使用命令行参数：
```bash
python run_UnlearningTask.py --config config_pretrain.yaml --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedAvg --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 2000 --E 1 --lr 0.05 --decay 0.999 --step_type bgd --unlearn_cn 1 --unlearn_pretrain True --save_model True
```

2) 运行 Fed-GMM 遗忘（最简单的方式）
```bash
python run_UnlearningTask.py
```

如果没有指定 `--config` 参数，程序会自动查找并使用 `config_unlearning.yaml` 配置文件。

你也可以显式指定配置文件：
```bash
python run_UnlearningTask.py --config config_unlearning.yaml
```

### 方式二：使用命令行参数（不使用配置文件）

1) 预训练全局模型（保持原有联邦训练不变）
```bash
python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedAvg --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 2000 --E 1 --lr 0.05 --decay 0.999 --step_type bgd --unlearn_cn 1 --unlearn_pretrain True --save_model True
```

2) 运行 Fed-GMM 遗忘
```bash
python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedGMM --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 2000 --UR 2000 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --lr 0.001 --gmm_rho 0.2 --gmm_mask_refresh 10 --gmm_kl_weight 5.0 --gmm_temperature 2.0 --early_stop True --early_stop_threshold 0.05 --early_stop_patience 10 --early_stop_min_delta 0.001
```

## 关键参数
- `gmm_rho`：掩码稀疏率（Top-ρ 参数被更新，默认 0.05）
- `gmm_mask_refresh`：掩码刷新间隔（0 表示仅初次计算）
- `gmm_kl_weight`：KL-to-uniform 损失权重
- `gmm_temperature`：KL 温度
- `UR` / `r_lr`：遗忘阶段轮数与恢复阶段学习率
- `early_stop`：是否启用早停（默认 True）
- `early_stop_threshold`：ASR 阈值，当 ASR <= threshold 时停止
- `early_stop_patience`：早停耐心值，连续 N 轮无改善则停止
- `early_stop_min_delta`：视为改善的最小 ASR 变化量

## 配置文件说明

项目支持使用 YAML 配置文件来管理参数。配置文件优先级：**命令行参数 > 配置文件 > 默认值**

### 配置文件列表

- **`config_unlearning.yaml`**：遗忘任务配置文件（默认配置文件）
  - 直接运行 `python run_UnlearningTask.py` 时会自动使用此文件
  - 包含 FedGMM 遗忘算法的所有参数

- **`config_pretrain.yaml`**：联邦学习预训练配置文件
  - 用于预训练全局模型
  - 使用方式：`python run_UnlearningTask.py --config config_pretrain.yaml`

### 配置文件使用

你可以：
- 直接编辑配置文件修改参数
- 使用 `--config` 参数指定配置文件路径
- 在命令行中覆盖配置文件中的特定参数（如 `--lr 0.002` 会覆盖配置文件中的 `lr` 值）
- 如果不指定 `--config`，遗忘任务会自动查找并使用 `config_unlearning.yaml`

## 流程简述
- 探测：对遗忘客户端做一次梯度，按幅值选 Top-ρ 生成掩码
- 清洗：遗忘阶段仅更新掩码区域，使用 KL-to-uniform 拉平输出
- 聚合：依然按数据量加权聚合；恢复阶段回退为标准 FedAvg 仅在保留客户端上训练

## 输出
日志路径与命名规则保持不变（同 `UnlearningTask` 默认行为），保存的模型与 log 文件位于 `UnlearningTask/<module>/<dataloader>/.../FedGMM/` 目录。
