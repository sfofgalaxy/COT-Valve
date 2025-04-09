#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import os
import json
from datetime import datetime

def run_command(cmd, description=None):
    """运行shell命令并打印输出"""
    if description:
        print(f"\n{'='*10} {description} {'='*10}")
    
    print(f"执行命令: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # 实时打印输出
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    return process.returncode

def train_model(args):
    """训练模型的函数"""
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 记录训练参数
    config = {
        "model": args.model,
        "dataset": args.dataset,
        "strategy": args.strategy,
        "output_dir": args.output_dir,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # 构建训练命令
    cmd = [
        "python", "train.py",
        "--model", args.model,
        "--dataset", args.dataset,
        "--strategy", args.strategy,
        "--output_dir", args.output_dir
    ]
    
    # 运行训练
    return run_command(cmd, "训练CoT-Valve模型")

def run_inference(args):
    """运行模型推理的函数"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # 如果是从训练目录推理，使用其中的final_checkpoint
    lora_path = args.lora_path
    if os.path.isdir(lora_path) and not lora_path.endswith("final_checkpoint"):
        if os.path.exists(os.path.join(lora_path, "final_checkpoint")):
            lora_path = os.path.join(lora_path, "final_checkpoint")
        elif args.strategy == "valve_plus_p" and os.path.exists(os.path.join(lora_path, "stage_5", "final_checkpoint")):
            lora_path = os.path.join(lora_path, "stage_5", "final_checkpoint")
    
    # 构建推理命令
    cmd = [
        "python", "inference.py",
        "--model", args.model,
        "--lora_path", lora_path,
        "--alpha", str(args.alpha),
        "--question", args.question
    ]
    
    if args.output_file:
        cmd.extend(["--output_file", args.output_file])
    
    # 运行推理
    return run_command(cmd, "运行CoT-Valve推理")

def run_length_sweep(args):
    """以不同的alpha值运行模型并比较结果"""
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定LoRA路径
    lora_path = args.lora_path
    if os.path.isdir(lora_path) and not lora_path.endswith("final_checkpoint"):
        if os.path.exists(os.path.join(lora_path, "final_checkpoint")):
            lora_path = os.path.join(lora_path, "final_checkpoint")
        elif args.strategy == "valve_plus_p" and os.path.exists(os.path.join(lora_path, "stage_5", "final_checkpoint")):
            lora_path = os.path.join(lora_path, "stage_5", "final_checkpoint")
    
    # 尝试不同的alpha值
    alpha_values = [0.2, 0.5, 1.0, 1.5, 2.0]
    results = []
    
    for alpha in alpha_values:
        # 为每个alpha值设置输出文件
        alpha_output_file = args.output_file.replace(".txt", f"_alpha_{alpha}.txt")
        
        # 构建命令
        cmd = [
            "python", "inference.py",
            "--model", args.model,
            "--lora_path", lora_path,
            "--alpha", str(alpha),
            "--question", args.question,
            "--output_file", alpha_output_file
        ]
        
        # 运行命令
        print(f"\n{'='*10} 使用alpha={alpha}运行推理 {'='*10}")
        run_command(cmd)
        
        # 读取生成的结果并记录token数量
        with open(alpha_output_file, "r", encoding="utf-8") as f:
            content = f.read()
            # 简单估计token数量（实际应该用tokenizer计算）
            response_part = content.split("回答 (alpha=")[1].split(":", 1)[1].strip()
            token_count = len(response_part.split())
            results.append((alpha, token_count, alpha_output_file))
    
    # 创建比较报告
    report_file = os.path.join(output_dir, "alpha_sweep_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"问题: {args.question}\n\n")
        f.write("Alpha值对CoT长度的影响:\n")
        f.write("-" * 50 + "\n")
        f.write("| Alpha值 | 估计Token数 | 输出文件 |\n")
        f.write("|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 30 + "|\n")
        
        for alpha, tokens, file in sorted(results):
            f.write(f"| {alpha:7.1f} | {tokens:10d} | {os.path.basename(file)} |\n")
    
    print(f"\n报告已生成: {report_file}")
    return 0

def main():
    parser = argparse.ArgumentParser(description="CoT-Valve训练和推理脚本")
    subparsers = parser.add_subparsers(dest="command", help="选择操作")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练CoT-Valve模型")
    train_parser.add_argument("--model", type=str, default="qwq_32b", 
                             choices=["qwq_32b", "deepseek_r1_32b", "deepseek_r1_14b"], 
                             help="要微调的模型")
    train_parser.add_argument("--dataset", type=str, default="mixchain_z_gsm8k", 
                             choices=["mixchain_z_gsm8k", "mixchain_z_prm12k", "mixchain_c_limo"], 
                             help="训练数据集")
    train_parser.add_argument("--strategy", type=str, default="standard", 
                             choices=["standard", "valve_plus_plus", "valve_plus_p"], 
                             help="训练策略")
    train_parser.add_argument("--output_dir", type=str, default="./cot_valve_output", 
                             help="模型输出目录")
    
    # 推理命令
    infer_parser = subparsers.add_parser("infer", help="使用CoT-Valve模型进行推理")
    infer_parser.add_argument("--model", type=str, default="qwq_32b", 
                             choices=["qwq_32b", "deepseek_r1_32b", "deepseek_r1_14b"], 
                             help="使用的基础模型")
    infer_parser.add_argument("--lora_path", type=str, required=True, 
                             help="训练好的LoRA适配器目录路径")
    infer_parser.add_argument("--alpha", type=float, default=1.0, 
                             help="LoRA的alpha缩放因子（控制CoT长度）")
    infer_parser.add_argument("--question", type=str, required=True, 
                             help="向模型提问的问题")
    infer_parser.add_argument("--output_file", type=str, default="./results/output.txt", 
                             help="保存输出到文件")
    infer_parser.add_argument("--strategy", type=str, default="standard", 
                             help="训练策略（用于确定正确的checkpoint路径）")
    
    # 长度扫描命令
    sweep_parser = subparsers.add_parser("sweep", help="用不同alpha值进行推理并比较结果")
    sweep_parser.add_argument("--model", type=str, default="qwq_32b", 
                              choices=["qwq_32b", "deepseek_r1_32b", "deepseek_r1_14b"], 
                              help="使用的基础模型")
    sweep_parser.add_argument("--lora_path", type=str, required=True, 
                              help="训练好的LoRA适配器目录路径")
    sweep_parser.add_argument("--question", type=str, required=True, 
                              help="向模型提问的问题")
    sweep_parser.add_argument("--output_file", type=str, default="./results/sweep_output.txt", 
                              help="保存输出到文件")
    sweep_parser.add_argument("--strategy", type=str, default="standard", 
                              help="训练策略（用于确定正确的checkpoint路径）")
    
    args = parser.parse_args()
    
    if args.command == "train":
        return train_model(args)
    elif args.command == "infer":
        return run_inference(args)
    elif args.command == "sweep":
        return run_length_sweep(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    exit(main()) 