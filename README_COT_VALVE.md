# COT-Valve 实现

这个代码库实现了论文中提出的CoT-Valve方法，该方法能够通过一个参数控制Chain-of-Thought推理链的长度。

## 简介

CoT-Valve是一种新的微调和推理策略，允许模型生成不同长度的推理链，从而动态平衡推理质量和计算开销。

主要特点：
- 在参数空间中找到控制CoT长度的方向（通过LoRA实现）
- 支持使用α参数在推理时控制CoT长度：
  - α < 1：生成更长的推理链（更详细的解释）
  - α = 1：标准CoT推理链
  - α > 1：生成更短的推理链（更高效）
- 增强策略：
  - CoT-Valve++：使用长度因子β进行更精确的长度控制
  - CoT-Valve+P：渐进式训练方法，从长推理链逐步过渡到短推理链

## 安装

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30.0+
- PEFT 0.5.0+
- Datasets
- TRL

安装依赖：

```bash
pip install torch>=2.0 transformers>=4.30.0 peft>=0.5.0 datasets accelerate trl
```

## 使用方法

### 数据集

支持以下MixChain数据集：
- MixChain-Z-GSM8K：包含6,863个样本，每个样本有5个不同长度的解答
- MixChain-Z-PRM12K：包含12,000个样本，每个样本有5个不同长度的解答
- MixChain-C-LIMO：LIMO数据集的变体，每个问题包含两种不同长度的解答

### 训练模型

使用`run.py`脚本进行训练：

```bash
python run.py train --model qwq_32b --dataset mixchain_z_gsm8k --strategy standard --output_dir ./outputs/qwq_valve
```

参数说明：
- `--model`：选择基础模型（可选：qwq_32b, deepseek_r1_32b, deepseek_r1_14b）
- `--dataset`：选择训练数据集（可选：mixchain_z_gsm8k, mixchain_z_prm12k, mixchain_c_limo）
- `--strategy`：训练策略（可选：standard, valve_plus_plus, valve_plus_p）
- `--output_dir`：输出目录

### 推理

使用训练好的模型进行推理：

```bash
python run.py infer --model qwq_32b --lora_path ./outputs/qwq_valve --alpha 0.5 --question "John有5个苹果。他又买了3个。他现在有多少个苹果？"
```

参数说明：
- `--model`：选择基础模型
- `--lora_path`：LoRA适配器路径
- `--alpha`：控制CoT长度的参数（<1生成更长解释，>1生成更简洁解释）
- `--question`：问题文本
- `--output_file`：（可选）输出文件路径

### Alpha值效果测试

测试不同alpha值对CoT长度的影响：

```bash
python run.py sweep --model qwq_32b --lora_path ./outputs/qwq_valve --question "John有5个苹果。他又买了3个。他现在有多少个苹果？"
```

这将使用一系列不同的alpha值进行推理，并生成比较报告。

## 高级用法

### CoT-Valve++

使用beta值进行更精确的长度控制：

```bash
python run.py train --model qwq_32b --dataset mixchain_z_gsm8k --strategy valve_plus_plus --output_dir ./outputs/qwq_valve_plus_plus
```

### CoT-Valve+P

使用渐进式训练逐步压缩CoT长度：

```bash
python run.py train --model qwq_32b --dataset mixchain_z_gsm8k --strategy valve_plus_p --output_dir ./outputs/qwq_valve_plus_p
```

## 自定义配置

可以在`config.py`中修改以下设置：
- 模型路径
- LoRA参数（rank, alpha, dropout等）
- 训练参数（epochs, batch size, learning rate等）
- 生成参数（max_new_tokens, temperature等）

## 项目结构

```
.
├── config.py             # 配置文件
├── data_utils.py         # 数据处理工具
├── inference.py          # 推理脚本
├── model_utils.py        # 模型工具函数
├── run.py                # 主运行脚本
├── train.py              # 训练脚本
└── README_COT_VALVE.md   # 本文档
```

## 论文引用

如果您使用了本代码，请引用原论文：

```
@article{CoT-Valve,
  title={CoT-Valve: A Unified Framework for Controllable Chain-of-Thought Reasoning},
  author={...},
  journal={...},
  year={2024}
}
``` 