import torch
from transformers import GenerationConfig
import argparse
import os
from config import MODEL_CONFIG, DATA_CONFIG, GENERATION_CONFIG, MODEL_PATHS
from data_utils import format_prompt, get_tokenizer
from model_utils import load_model_for_inference

def run_inference(model, text_input, alpha_scale, tokenizer):
    """
    使用加载的模型（可能带有特定alpha的LoRA）进行推理
    
    Args:
        model: 加载的模型
        text_input: 问题文本
        alpha_scale: LoRA缩放因子
        tokenizer: 分词器
    
    Returns:
        生成的回答文本
    """
    prompt = format_prompt({"question": text_input}, solution_index=None)  # 获取用于推理的prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    generation_config = GenerationConfig(
        max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
        do_sample=GENERATION_CONFIG["do_sample"],
        temperature=GENERATION_CONFIG["temperature"],
        top_p=GENERATION_CONFIG["top_p"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    print(f"使用alpha_scale = {alpha_scale}生成回答...")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)
    
    # 解码生成的token IDs，跳过输入的prompt部分
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return generated_text.strip()

def main():
    parser = argparse.ArgumentParser(description="运行CoT-Valve推理。")
    parser.add_argument("--model", type=str, default="qwq_32b", choices=["qwq_32b", "deepseek_r1_32b", "deepseek_r1_14b"], 
                        help="使用的基础模型")
    parser.add_argument("--lora_path", type=str, required=True, help="训练好的LoRA适配器目录路径。")
    parser.add_argument("--alpha", type=float, default=1.0, help="LoRA的alpha缩放因子（控制CoT长度）。")
    parser.add_argument("--question", type=str, required=True, help="向模型提问的问题。")
    parser.add_argument("--output_file", type=str, default=None, help="保存输出到文件（可选）。")
    args = parser.parse_args()
    
    # 获取模型路径
    model_path = MODEL_PATHS[args.model]
    
    # 1. 加载tokenizer
    tokenizer = get_tokenizer(model_path)
    
    # 2. 加载带LoRA的模型，应用alpha缩放
    model = load_model_for_inference(
        model_path=model_path,
        lora_adapter_path=args.lora_path, 
        alpha_scale=args.alpha
    )
    
    # 3. 运行推理
    response = run_inference(model, args.question, args.alpha, tokenizer)
    
    # 4. 输出结果
    print("\n" + "="*20 + " 模型回答 " + "="*20)
    print(response)
    print("="*50)
    
    # 5. 如果需要，保存到文件
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f"问题: {args.question}\n\n")
            f.write(f"回答 (alpha={args.alpha}):\n{response}")
        print(f"回答已保存到: {args.output_file}")

if __name__ == "__main__":
    main()

# 示例运行:
# python inference.py --model qwq_32b --lora_path ./cot_valve_output/qwq_32b_mixchain_z_gsm8k_standard/final_checkpoint --alpha 0.5 --question "John有5个苹果。他又买了3个。他现在有多少个苹果？"
# python inference.py --model qwq_32b --lora_path ./cot_valve_output/qwq_32b_mixchain_z_gsm8k_standard/final_checkpoint --alpha 1.5 --question "John有5个苹果。他又买了3个。他现在有多少个苹果？"