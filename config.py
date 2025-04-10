# 调试模式配置
DEBUG_MODE = False  # 默认为调试模式

# 根据调试模式选择模型路径
if DEBUG_MODE:
    MODEL_PATHS = {
        "qwq_32b": "/data_sde/lyl/QwQ-32B-AWQ",
        "deepseek_r1_32b": "/data_sde/lyl/deepseek-r1-distill-qwen-32b-awq",
        "deepseek_r1_14b": "/data_sde/lyl/deepseek-r1-distill-qwen-14b-awq"
    }
else:
    MODEL_PATHS = {
        "qwq_32b": "~/data/model/QwQ-32B-AWQ",
        "deepseek_r1_32b": "~/data/model/deepseek-r1-distill-qwen-32b-awq",
        "deepseek_r1_14b": "~/data/model/deepseek-r1-distill-qwen-14b-awq"
    }

MODEL_CONFIG = {
    "base_model_path": MODEL_PATHS["qwq_32b"], # 默认模型，可通过命令行参数更改
    "output_dir": "./cot_valve_output",
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 适用于QwQ和deepseek模型
}
DATA_CONFIG = {
    "train_dataset_path": "horseee/MixChain-Z-GSM8K", # 默认使用MixChain-Z-GSM8K
    "eval_dataset_path": "gsm8k", # 评估数据集
    "dataset_config_name": "default", # 使用default配置
    "question_column": "question",
    "solution_column_prefix": "solution_", # MixChain数据集中的solution列名格式
    "max_seq_length": 2048,
}
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2, # 量化模型可能需要更小的批大小
    "gradient_accumulation_steps": 16, # 增加梯度累积
    "learning_rate": 5e-5, # 对于LoRA微调
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 200,
    "fp16": True, # 如果硬件支持
    "bf16": False, # 一些新GPU支持bf16精度
    "seed": 42,
    "use_valve_plus_plus": False, # 是否使用CoT-Valve++策略
    "use_valve_plus_p": False, # 是否使用CoT-Valve+P策略
    "valve_plus_p_epochs_per_stage": 2, # CoT-Valve+P每个阶段的epoch数
}
GENERATION_CONFIG = {
     "max_new_tokens": 1024, # 根据任务调整
     "do_sample": False, # 论文中使用greedy decoding
     "temperature": 1.0, # 如果do_sample=True
     "top_p": 1.0,       # 如果do_sample=True
}

DATASET_PATHS = {
    "mixchain_z_gsm8k": "horseee/MixChain-Z-GSM8K",
    "mixchain_z_prm12k": "horseee/MixChain-Z-PRM12K",
    "mixchain_c_limo": "horseee/MixChain-C-LIMO"
}