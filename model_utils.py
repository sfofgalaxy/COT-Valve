from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import torch
from config import MODEL_CONFIG
def load_model_for_training(model_path=None):
    if model_path is None:
        model_path = MODEL_CONFIG["base_model_path"]
    
    print(f"Loading base model from: {model_path}")
    
    # 对于AWQ量化模型的配置
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载量化模型，AWQ量化模型通常会自动检测并加载
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,  # 使用半精度
    )
    
    # 准备模型用于LoRA训练
    model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=MODEL_CONFIG["lora_rank"],
        lora_alpha=MODEL_CONFIG["lora_alpha"],
        target_modules=MODEL_CONFIG["target_modules"],
        lora_dropout=MODEL_CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 将LoRA应用到模型
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数信息
    
    return model
def load_model_for_inference(model_path=None, lora_adapter_path=None, alpha_scale=1.0):
    # 加载基础模型
    if model_path is None:
        model_path = MODEL_CONFIG["base_model_path"]
    
    print(f"Loading base model from: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    if lora_adapter_path:
        print(f"Loading LoRA adapter from: {lora_adapter_path} with scale: {alpha_scale}")
        
        # 加载LoRA适配器
        model = PeftModel.from_pretrained(model, lora_adapter_path, adapter_name="cot_valve")
        
        # 设置LoRA缩放比例
        # PEFT 0.5.0+版本支持设置适配器缩放
        try:
            # 对于较新的PEFT版本
            model.set_adapter_scale("cot_valve", alpha_scale)
            print(f"Successfully set adapter scale to {alpha_scale}")
        except Exception as e:
            print(f"Warning: Could not set adapter scale directly: {e}")
            print("Attempting alternative scaling method...")
            
            # 备用方法：手动设置LoRA层的缩放
            for name, module in model.named_modules():
                if "lora" in name.lower() and hasattr(module, "scaling"):
                    original_scaling = module.scaling.get("cot_valve", 1.0)
                    module.scaling["cot_valve"] = original_scaling * alpha_scale
                    print(f"Manually set scaling for {name} to {module.scaling['cot_valve']}")
        
        # 激活适配器
        model.enable_adapters()
    
    model.eval()  # 设置为评估模式
    return model
def create_valve_plus_plus_model(model, beta_values):
    """
    为CoT-Valve++创建模型
    
    Args:
        model: 基础模型
        beta_values: 长度因子β值列表
    
    Returns:
        model: 配置用于CoT-Valve++训练的模型
    """
    # 创建针对不同beta值的模型，这里简化为使用同一个模型
    # 实际实现可能需要更复杂的逻辑来处理beta
    
    # 把beta值保存到模型中，以便训练时可以访问
    model.beta_values = beta_values
    
    return model