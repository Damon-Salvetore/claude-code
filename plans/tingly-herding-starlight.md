# NVFP4 实验训练脚本 & 监控 & 数据收集方案

## Context

NVFP4 W4A4 量化实验需要跑 8 个实验（4 模型架构 × baseline/nvfp4），每个实验用 4×B200 180GB 单机跑，多台机器并行。
代码实现已全部就绪（nvfp4_quant.py、ffn.py、model.py、config.py），现在需要配套的运维脚本。

**核心需求：**
- 不需要存 checkpoint（`--no_save`），只看收敛 loss/BPB、grad_norm、throughput
- 用 wandb 收集和可视化所有指标
- 一键启动 + 实时监控 + 训练完成后汇总对比

## 需要创建的文件

### 1. `scripts/nvfp4_experiment/run_single.sh` — 单实验启动脚本
- 接受参数：`MODEL_NAME`、`MODE`（baseline/nvfp4）
- 内部调用 `torchrun --nproc_per_node=4 llm/nnscaler_train.py`
- 关键参数：
  - `--model $MODEL --data debug --hyperparams nvfp4_experiment`
  - `--no_save`（不存 checkpoint）
  - `--batch_size 2 --update_freq 2 --max_seq_len 4096 --seed 42`
  - `--plan_ngpus 1 --runtime_ngpus 4 --zero_group_size 4`
  - `--precision fp32 --xentropy_recompute --use_async_reducer --disable_shared_param_constraint`
  - `--gpu_mem_constraint 135`
  - `--learning_rate 6e-4 --min_lr 3e-5 --warmup_iters 2000 --max_updates 10000`
  - `--wandb_project nvfp4_experiment --wandb_entity 3259482542-south-china-university-of-technology`
  - baseline: `--nvfp4_mode 0`，nvfp4: `--nvfp4_mode 1`
  - `--name $MODEL-$MODE`（wandb run name）
  - `--save_path /tmp/nvfp4_runs/$MODEL-$MODE`（wandb runs 目录，仅放日志）
- stdout/stderr 重定向到 `logs/$MODEL-$MODE.log`

### 2. `scripts/nvfp4_experiment/run_all.sh` — 批量启动脚本
- 循环 4 个模型 × 2 种模式 = 8 个实验
- 每个实验调用 `run_single.sh`，后台运行（`&`），记录 PID
- 生成 `logs/pids.txt` 记录所有实验的 PID 和名称
- 设计为在单台机器上串行跑（一次一个，因为每个都占满 4 卡），或手动在不同机器上各跑一个
- 提供两种模式：`--parallel`（全部后台并行，用于多机）和默认串行（单机依次跑）

### 3. `scripts/nvfp4_experiment/monitor.sh` — 实时监控脚本
- 检查所有实验进程是否存活（基于 pids.txt）
- 从每个实验的日志文件中 tail 最近的 loss/gnorm/lr/throughput
- 显示各实验的当前 step、ETA
- 刷新显示

### 4. `scripts/nvfp4_experiment/collect_results.py` — 结果收集脚本
- 从 wandb API 拉取 8 个 run 的 metrics
- 或者从本地日志文件解析 train_loss、gnorm、lr、throughput
- 输出对比表格：
  - 每个模型的 baseline vs nvfp4 最终 loss
  - loss 曲线对比数据（CSV 导出）
  - grad_norm 稳定性对比

## 关键文件路径
- `llm/nnscaler_train.py` — 训练入口，已有完整的 wandb 集成
- `llm/config.py` — 已注册好 model_args、training_args、data_args
- wandb entity: `3259482542-south-china-university-of-technology`
- wandb project: `nvfp4_experiment`

## nnscaler_train.py 自动记录的指标（不需要额外改代码）
| 指标 | wandb key | 说明 |
|------|-----------|------|
| Loss (BPB) | `train/train_loss` | bits per byte，核心对比指标 |
| Grad norm | `train/gnorm` | 梯度范数（裁剪前） |
| Learning rate | `train/lr` | 学习率 |
| Throughput | `train/throughput` | tokens/s |
| Z-loss | `train/z_loss` | 默认 0（本实验不启用） |
| GPU memory | `train/cuda_gb_allocated` | 显存使用 |
| Step wall time | `train/train_wall` | 每步耗时 |
| FLOPS | `train/FLOPS` | 每 GPU FLOPS |

## 验证方式
1. 先用 `run_single.sh Base-0.4B baseline` 跑一个验证脚本能否正常启动
2. 检查 wandb 面板是否有 metrics 上报
3. 确认 `--no_save` 不产生 checkpoint 文件
4. 然后 `run_all.sh` 批量启动全部实验
