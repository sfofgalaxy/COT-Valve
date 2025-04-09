import transformers
from transformers import TrainingArguments
import torch
import os
import argparse
from trl import SFTTrainer
from config import MODEL_CONFIG, DATA_CONFIG, TRAINING_CONFIG, MODEL_PATHS, DATASET_PATHS
from data_utils import load_and_prepare_data
from model_utils import load_model_for_training, create_valve_plus_plus_model

def train_valve_plus_plus(model, dataset, training_args, tokenizer):
    """
    使用CoT-Valve++策略训练模型。
    这种方法将beta值作为额外的输入因子，明确控制CoT长度。
    
    Args:
        model: 基础模型
        dataset: 包含beta值的数据集
        training_args: 训练参数
        tokenizer: 分词器
    
    Returns:
        训练好的模型
    """
    # 提取数据集中的beta值
    beta_values = set()
    for item in dataset:
        if "beta" in item:
            beta_values.add(item["beta"])
    
    beta_values = sorted(list(beta_values))
    print(f"训练数据集中的beta值: {beta_values}")
    
    # 准备带有beta值的模型
    model = create_valve_plus_plus_model(model, beta_values)
    
    # 定义beta取样函数，用于训练过程
    def beta_sampling_callback(model, batch):
        # 获取当前batch中的beta值
        betas = [example.get("beta", 0.0) for example in batch]
        # 将beta值存储在模型中，供前向传播使用
        model.current_beta_values = betas
        return batch
    
    # 创建SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=DATA_CONFIG["max_seq_length"],
        tokenizer=tokenizer,
        packing=False,
        callbacks=[beta_sampling_callback]
    )
    
    # 开始训练
    print("开始CoT-Valve++训练...")
    trainer.train()
    
    return trainer

def train_valve_plus_p(model_path, dataset_path, output_dir, tokenizer):
    """
    使用CoT-Valve+P渐进式训练策略。
    这种方法逐步从长CoT链切换到短CoT链。
    
    Args:
        model_path: 模型路径
        dataset_path: 数据集路径
        output_dir: 输出目录
        tokenizer: 分词器
    
    Returns:
        训练好的模型路径
    """
    stages = 5  # 总共分为5个阶段，从长到短
    epochs_per_stage = TRAINING_CONFIG["valve_plus_p_epochs_per_stage"]
    
    # 初始化模型
    model = load_model_for_training(model_path)
    
    # 渐进式训练：从长到短
    for stage in range(stages):
        print(f"\n开始CoT-Valve+P阶段 {stage+1}/{stages}")
        
        # 为当前阶段准备数据集
        dataset = load_and_prepare_data(
            model_path=model_path,
            dataset_path=dataset_path,
            for_training=True,
            strategy="valve_plus_p",
            stage=stage
        )
        
        # 当前阶段的输出目录
        stage_output_dir = os.path.join(output_dir, f"stage_{stage+1}")
        os.makedirs(stage_output_dir, exist_ok=True)
        
        # 设置当前阶段的训练参数
        training_args = TrainingArguments(
            output_dir=stage_output_dir,
            num_train_epochs=epochs_per_stage,
            per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
            lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
            warmup_steps=TRAINING_CONFIG["warmup_steps"],
            logging_dir=f"{stage_output_dir}/logs",
            logging_steps=TRAINING_CONFIG["logging_steps"],
            save_strategy="steps",
            save_steps=TRAINING_CONFIG["save_steps"],
            save_total_limit=2,
            fp16=TRAINING_CONFIG["fp16"],
            bf16=TRAINING_CONFIG["bf16"],
            report_to="tensorboard",
            seed=TRAINING_CONFIG["seed"],
        )
        
        # 创建SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=DATA_CONFIG["max_seq_length"],
            tokenizer=tokenizer,
            packing=False,
        )
        
        # 训练当前阶段
        trainer.train()
        
        # 保存当前阶段的模型
        trainer.save_model(os.path.join(stage_output_dir, "final_checkpoint"))
        
        # 如果不是最后一个阶段，为下一阶段重新加载模型
        if stage < stages - 1:
            model = load_model_for_training(model_path)
            model.load_adapter(os.path.join(stage_output_dir, "final_checkpoint"))
    
    # 返回最终阶段的模型路径
    final_checkpoint_dir = os.path.join(output_dir, f"stage_{stages}", "final_checkpoint")
    return final_checkpoint_dir

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="训练CoT-Valve模型。")
    parser.add_argument("--model", type=str, default="qwq_32b", choices=["qwq_32b", "deepseek_r1_32b", "deepseek_r1_14b"], 
                        help="要微调的模型")
    parser.add_argument("--dataset", type=str, default="mixchain_z_gsm8k", 
                        choices=["mixchain_z_gsm8k", "mixchain_z_prm12k", "mixchain_c_limo"], 
                        help="训练数据集")
    parser.add_argument("--strategy", type=str, default="standard", 
                        choices=["standard", "valve_plus_plus", "valve_plus_p"], 
                        help="训练策略")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="模型输出目录")
    args = parser.parse_args()
    
    # 根据命令行参数获取模型路径和数据集路径
    model_path = MODEL_PATHS[args.model]
    dataset_path = DATASET_PATHS[args.dataset]
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(
            MODEL_CONFIG["output_dir"], 
            f"{args.model}_{args.dataset}_{args.strategy}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"模型: {args.model} ({model_path})")
    print(f"数据集: {args.dataset} ({dataset_path})")
    print(f"策略: {args.strategy}")
    print(f"输出目录: {args.output_dir}")
    
    # 1. 加载tokenizer
    from data_utils import get_tokenizer
    tokenizer = get_tokenizer(model_path)
    
    # 2. 根据策略决定训练方法
    if args.strategy == "valve_plus_p":
        # CoT-Valve+P：渐进式训练
        final_checkpoint_dir = train_valve_plus_p(
            model_path, dataset_path, args.output_dir, tokenizer
        )
        print(f"CoT-Valve+P训练完成。最终模型保存在: {final_checkpoint_dir}")
    else:
        # 标准SFT或CoT-Valve++
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
            per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
            lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
            warmup_steps=TRAINING_CONFIG["warmup_steps"],
            logging_dir=f"{args.output_dir}/logs",
            logging_steps=TRAINING_CONFIG["logging_steps"],
            save_strategy="steps",
            save_steps=TRAINING_CONFIG["save_steps"],
            save_total_limit=3,
            fp16=TRAINING_CONFIG["fp16"],
            bf16=TRAINING_CONFIG["bf16"],
            report_to="tensorboard",
            seed=TRAINING_CONFIG["seed"],
        )
        
        # 1. 加载模型
        model = load_model_for_training(model_path)
        
        # 2. 加载数据
        strategy_for_data = args.strategy if args.strategy == "valve_plus_plus" else None
        train_dataset = load_and_prepare_data(
            model_path=model_path,
            dataset_path=dataset_path,
            for_training=True,
            strategy=strategy_for_data
        )
        
        # 3. 根据策略选择训练方法
        if args.strategy == "valve_plus_plus":
            # CoT-Valve++训练
            trainer = train_valve_plus_plus(model, train_dataset, training_args, tokenizer)
        else:
            # 标准SFT训练
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                dataset_text_field="text",
                max_seq_length=DATA_CONFIG["max_seq_length"],
                tokenizer=tokenizer,
                packing=False,
            )
            
            print("开始标准SFT训练...")
            trainer.train()
        
        # 4. 保存最终模型
        final_checkpoint_dir = f"{args.output_dir}/final_checkpoint"
        trainer.save_model(final_checkpoint_dir)
        tokenizer.save_pretrained(final_checkpoint_dir)
        print(f"训练完成。最终模型保存在: {final_checkpoint_dir}")

if __name__ == "__main__":
    main()