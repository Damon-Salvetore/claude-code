# 实现计划：MXFP4 (W4) + MXFP8 (A8) 量化模式

## Context

在 B200 (Blackwell) 上，真正的 W4A8 硬件加速路径要求 weight 和 activation 都使用 OCP MX 格式（block=32, E8M0 scale）。现有项目已有 NVFP4 W4A4 fake quantization（`nvfp4_mode=1/2`），但那条路径在 tensor core 上实际是 W4A4（activation 也被量化到 FP4）。

本任务新增 MXFP4+MXFP8 量化模式，让训练感知真正的 W4A8 量化噪声，为 Blackwell 推理做 QAT 准备。

## 方案概述

新增 `nvfp4_mode=3`：MXFP4 (weight) + MXFP8 (activation) 全量化（FFN + QKVO），与现有 W4A4 的 mode=1/2 区分开。

### MX 格式规格对比

| | NVFP4 (现有 mode=1/2) | MXFP4 weight (新 mode=3) | MXFP8 activation (新 mode=3) |
|---|---|---|---|
| 元素格式 | FP4 E2M1 | FP4 E2M1 | FP8 E4M3 |
| Block size | 16 | **32** | **32** |
| Scale format | E4M3 (FP8) | **E8M0 (2^n only)** | **E8M0 (2^n only)** |
| 二级 scale | 有 (per-tensor FP32) | **无** | **无** |
| 量化等级数 | 16 (8正+8负) | 16 (8正+8负) | 448级 (FP8 E4M3) |

## 实现步骤

### Step 1: 创建 `llm/kernel/mxfp_quant.py`

新文件，实现两个 fake quant 函数：

**`mxfp4_fake_quant(x)`** — MXFP4 weight 量化：
- Block size = 32
- Per-block amax → scale = amax / FP4_MAX (6.0)
- Scale 量化到 E8M0：`scale_e8m0 = 2^round(log2(scale))`（snap 到最近的 2 的幂次）
- 元素量化到 FP4 E2M1（复用 NVFP4 的值表和 searchsorted 逻辑）
- STE backward

**`mxfp8_fake_quant(x)`** — MXFP8 activation 量化：
- Block size = 32
- Per-block amax → scale = amax / FP8_E4M3_MAX (448.0)
- Scale 量化到 E8M0：`scale_e8m0 = 2^round(log2(scale))`
- 元素 cast 到 `torch.float8_e4m3fn` 再 cast 回来（和 NVFP4 的 scale quant 类似）
- STE backward

关键实现细节：
- E8M0 scale 量化：`torch.exp2(torch.round(torch.log2(scale.clamp(min=1e-12))))`
- FP4 E2M1 值表和 boundary 复用 `nvfp4_quant.py` 中已有的 `_FP4_VALUES` 和 `_FP4_BOUNDS`
- 使用 `@torch.compile` 优化
- Block size 常量：`MXFP_BLOCK_SIZE = 32`

### Step 2: 修改 `llm/arch/config.py`

在 `nvfp4_mode` 的注释中扩展说明：
```python
nvfp4_mode: int = 0
# 0: disable
# 1: FFN NVFP4 W4A4 fake quant
# 2: FFN+QKVO NVFP4 W4A4 fake quant
# 3: FFN+QKVO MXFP4(W4)+MXFP8(A8) fake quant (OCP MX format, Blackwell W4A8)
```

### Step 3: 修改 `llm/arch/ffn.py`

在 `FeedForwardNetwork.forward()` 中添加 `nvfp4_mode == 3` 分支：
- Weight 用 `mxfp4_fake_quant()`
- Activation 用 `mxfp8_fake_quant()`
- 保持与 mode=2 相同的量化位置（up_proj, gate_proj, down_proj 的 weight 和 input activation）

```python
if self.nvfp4_mode == 3:
    # MXFP4(W4) + MXFP8(A8): activation→MXFP8, weight→MXFP4
    x_fq = mxfp8_fake_quant(x)
    up = mix_precision_linear(x_fq, mxfp4_fake_quant(self.up_proj.weight))
    gate = mix_precision_linear(x_fq, mxfp4_fake_quant(self.gate_proj.weight))
    intermediate = swiglu(up, gate)
    inter_fq = mxfp8_fake_quant(intermediate)
    return mix_precision_linear(inter_fq, mxfp4_fake_quant(self.down_proj.weight))
```

### Step 4: 修改 `llm/arch/attention.py`

在 `Attention.forward()` 中添加 `nvfp4_mode == 3` 分支：
- 与 mode=2 相同的位置（Q/K/V/O projections），但用 MXFP4(weight) + MXFP8(activation)

### Step 5: 修改 `scripts/nvfp4_experiment/run_single.sh`

添加新 mode：
```bash
elif [[ "$MODE" == "mxw4a8" ]]; then
    NVFP4_FLAG="--nvfp4_mode 3"
```

### Step 6: 创建测试 `tests/test_mxfp_quant.py`

仿照 `tests/test_nvfp4_ffn.py` 的结构，测试：
1. E8M0 scale 量化正确性（snap 到 2^n）
2. MXFP4 value mapping（block=32）
3. MXFP8 fake quant 正确性
4. STE gradient pass-through
5. 多 block 量化
6. 2D tensor (nnscaler shape)
7. bf16 dtype 保持
8. 边界 case（all-zeros, very small values）
9. FFN forward/backward（nvfp4_mode=3）
10. FFN mode=3 vs baseline 有量化噪声
11. FFN mode=3 vs mode=2 输出不同（W4A8 vs W4A4 量化噪声不同）

### Step 7: 运行测试验证

```bash
cd /home/yingbohao/llm-train && python tests/test_mxfp_quant.py
```

确保所有测试通过后，再运行现有 NVFP4 测试确认没有 regression：
```bash
cd /home/yingbohao/llm-train && python tests/test_nvfp4_ffn.py
```

## 涉及的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `llm/kernel/mxfp_quant.py` | **新建** | MXFP4/MXFP8 fake quant 核心实现 |
| `llm/arch/config.py` | 修改 | nvfp4_mode 注释更新（加 mode=3 说明） |
| `llm/arch/ffn.py` | 修改 | 添加 mode=3 分支 |
| `llm/arch/attention.py` | 修改 | 添加 mode=3 分支 |
| `scripts/nvfp4_experiment/run_single.sh` | 修改 | 添加 mxw4a8 mode 映射 |
| `tests/test_mxfp_quant.py` | **新建** | MXFP4/MXFP8 量化测试 |
| `experiment_configuration_summary.md` | 修改 | 新增 MXFP4+MXFP8 实验配置章节 |

## 可复用的现有代码

- `llm/kernel/nvfp4_quant.py` 的 `_FP4_VALUES`, `_FP4_BOUNDS`, `_FP4_MAX` 常量 → MXFP4 的 FP4 E2M1 值表完全相同
- `llm/kernel/nvfp4_quant.py` 的 `NvFP4FakeQuant` class → STE autograd.Function 的模式
- `llm/arch/linear.py` 的 `mix_precision_linear` → GEMM 函数
- `tests/test_nvfp4_ffn.py` → 测试结构和模式

## 验证计划

1. **单元测试**：运行 `tests/test_mxfp_quant.py`，全部通过
2. **回归测试**：运行 `tests/test_nvfp4_ffn.py`，确认现有 NVFP4 不受影响
3. **端到端短训练验证**：`MAX_UPDATES=5` 跑一次 mode=3 训练，确认不崩溃、loss 正常下降
