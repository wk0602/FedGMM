# Fed-GMM: Federated Gradient Masking & Modification

轻量级联邦遗忘实现：用梯度敏感度生成稀疏掩码，仅更新最敏感的参数，并用 KL-to-uniform 让模型在待遗忘数据上“拉平”输出，减少对其余任务的伤害。

## 快速开始
1) 预训练全局模型（保持原有联邦训练不变）
```
/usr/bin/python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedAvg --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 2000 --E 1 --lr 0.05 --decay 0.999 --step_type bgd --unlearn_cn 1 --unlearn_pretrain True --save_model True
```

2) 运行 Fed-GMM 遗忘（替换原 FedOSD）
```
/usr/bin/python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedGMM --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --lr 0.0004 --r_lr 1e-6 --gmm_rho 0.05 --gmm_mask_refresh 0 --gmm_kl_weight 1.0 --gmm_temperature 1.0
```

## 关键参数
- `gmm_rho`：掩码稀疏率（Top-ρ 参数被更新，默认 0.05）
- `gmm_mask_refresh`：掩码刷新间隔（0 表示仅初次计算）
- `gmm_kl_weight`：KL-to-uniform 损失权重
- `gmm_temperature`：KL 温度
- `UR` / `r_lr`：遗忘阶段轮数与恢复阶段学习率

## 流程简述
- 探测：对遗忘客户端做一次梯度，按幅值选 Top-ρ 生成掩码
- 清洗：遗忘阶段仅更新掩码区域，使用 KL-to-uniform 拉平输出
- 聚合：依然按数据量加权聚合；恢复阶段回退为标准 FedAvg 仅在保留客户端上训练

## 输出
日志路径与命名规则保持不变（同 `UnlearningTask` 默认行为），保存的模型与 log 文件位于 `UnlearningTask/<module>/<dataloader>/.../FedGMM/` 目录。
