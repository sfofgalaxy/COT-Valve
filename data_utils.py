from datasets import load_dataset
from transformers import AutoTokenizer
import random
import numpy as np
from config import MODEL_CONFIG, DATA_CONFIG, TRAINING_CONFIG

# 初始化时不立即加载tokenizer，而是在需要时加载
tokenizer = None

# 定义DatasetWrapper类，用于包装处理后的数据集
class DatasetWrapper:
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i]
    
    def map(self, function, **kwargs):
        """
        实现map方法，与Hugging Face数据集接口兼容
        """
        # 检查函数是否需要特定参数
        import inspect
        sig = inspect.signature(function)
        
        # 创建一个包含所有可能需要的参数的字典
        func_kwargs = {}
        
        # 检查是否需要tokenizer参数
        if 'tokenizer' in sig.parameters:
            from data_utils import get_tokenizer
            func_kwargs['tokenizer'] = get_tokenizer()
        
        # 检查是否需要processing_class参数
        if 'processing_class' in sig.parameters:
            # 创建一个简单的处理函数，而不是一个类
            from data_utils import get_tokenizer
            tokenizer = get_tokenizer()
            eos_token_id = tokenizer.eos_token_id
            
            def simple_processor(text):
                # 返回一个包含input_ids键的字典，而不是一个字符串
                return {"input_ids": tokenizer.encode(text)}
            
            # 添加eos_token_id属性到函数
            simple_processor.eos_token_id = eos_token_id
            
            func_kwargs['processing_class'] = simple_processor
        
        # 检查是否需要dataset_text_field参数
        if 'dataset_text_field' in sig.parameters:
            func_kwargs['dataset_text_field'] = "text"
        
        # 应用函数到每个样本
        mapped_examples = [function(example, **func_kwargs) for example in self.examples]
        
        return DatasetWrapper(mapped_examples)

def get_tokenizer(model_path=None):
    global tokenizer
    if tokenizer is None:
        if model_path is None:
            model_path = MODEL_CONFIG["base_model_path"]
        print(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def format_prompt(example, solution_index=None):
    """
    格式化提示和回答。
    
    Args:
        example: 数据样本
        solution_index: 使用哪个解答（0, 1, 2, 3, 4等），None表示仅返回问题部分用于推理
    
    Returns:
        格式化后的文本
    """
    tok = get_tokenizer()
    question = example[DATA_CONFIG["question_column"]]
    
    if solution_index is None:
        # 用于推理，只返回问题部分
        return f"User: {question}\nAssistant:" 
    else:
        solution_key = f"{DATA_CONFIG['solution_column_prefix']}{solution_index}"
        if solution_key not in example or example[solution_key] is None:
            return None  # 跳过无效数据
        
        solution = example[solution_key]
        # SFT格式，包含问题和答案
        return f"User: {question}\nAssistant: {solution}{tok.eos_token}"

def calculate_token_length(text):
    """计算文本的token长度"""
    tok = get_tokenizer()
    return len(tok.encode(text))

def calculate_beta_values(examples):
    """
    计算CoT-Valve++的beta值（长度因子）
    
    Args:
        examples: 包含多个solution的样本列表
    
    Returns:
        beta_values: 每个solution的beta值词典，key为solution_index，值为beta值
    """
    all_lengths = {}
    
    # 1. 收集所有solution的长度
    for example in examples:
        valid_solutions = []
        for i in range(10):  # 假设最多10个solution
            solution_key = f"{DATA_CONFIG['solution_column_prefix']}{i}"
            if solution_key in example and example[solution_key] is not None:
                solution = example[solution_key]
                length = calculate_token_length(solution)
                if solution_key not in all_lengths:
                    all_lengths[solution_key] = []
                all_lengths[solution_key].append(length)
    
    # 2. 计算每种solution的平均长度
    avg_lengths = {k: np.mean(v) for k, v in all_lengths.items() if len(v) > 0}
    
    if not avg_lengths:
        print("警告：无法计算beta值，没有有效的solution")
        return {}
    
    # 3. 按照平均长度排序，最长的solution对应beta=0，最短的solution对应beta=1
    sorted_solutions = sorted(avg_lengths.items(), key=lambda x: x[1], reverse=True)
    
    # 如果只有一种solution，默认beta=0
    if len(sorted_solutions) == 1:
        return {sorted_solutions[0][0]: 0.0}
    
    max_length = sorted_solutions[0][1]  # 最长solution的长度
    min_length = sorted_solutions[-1][1]  # 最短solution的长度
    length_range = max_length - min_length
    
    # 4. 根据公式计算beta值
    beta_values = {}
    for solution_key, avg_length in sorted_solutions:
        if length_range > 0:
            # 归一化到[0,1]区间
            beta = 1.0 - (avg_length - min_length) / length_range
        else:
            beta = 0.0  # 如果所有solution长度相同，默认beta=0
        
        # 提取index部分，例如"solution_2" -> 2
        solution_index = int(solution_key.split("_")[-1])
        beta_values[solution_index] = beta
    
    return beta_values

def preprocess_dataset_for_sft(dataset, use_all_solutions=True, strategy=None):
    """
    预处理数据集用于监督微调。
    
    Args:
        dataset: 原始数据集
        use_all_solutions: 是否使用所有解答，False时只随机选择一个
        strategy: 训练策略，可选值有"valve_plus_plus"和"valve_plus_p"
    
    Returns:
        处理后的数据集
    """
    tok = get_tokenizer()
    processed_examples = []
    solution_keys = [col for col in dataset.column_names if col.startswith(DATA_CONFIG["solution_column_prefix"])]
    
    # 计算beta值，用于CoT-Valve++
    beta_values = None
    if strategy == "valve_plus_plus":
        beta_values = calculate_beta_values(dataset)
        print(f"Calculated beta values: {beta_values}")
    
    for example in dataset:
        if use_all_solutions:
            # 将每个solution作为独立的训练样本
            valid_solutions = []
            for i in range(len(solution_keys)):
                formatted = format_prompt(example, solution_index=i)
                if formatted:
                    item = {"text": formatted}
                    # 如果使用CoT-Valve++，添加beta值
                    if strategy == "valve_plus_plus" and i in beta_values:
                        item["beta"] = beta_values[i]
                    valid_solutions.append(item)
            
            # 如果使用CoT-Valve+P，按照solution的长度排序
            if strategy == "valve_plus_p":
                # 计算每个solution的长度并排序，从长到短
                valid_solutions = sorted(
                    valid_solutions, 
                    key=lambda x: calculate_token_length(x["text"]), 
                    reverse=True
                )
            
            processed_examples.extend(valid_solutions)
        else:
            # 只选择一个solution
            valid_indices = [
                i for i in range(len(solution_keys)) 
                if f"{DATA_CONFIG['solution_column_prefix']}{i}" in example 
                and example[f"{DATA_CONFIG['solution_column_prefix']}{i}"] is not None
            ]
            
            if valid_indices:
                chosen_index = random.choice(valid_indices)
                formatted = format_prompt(example, solution_index=chosen_index)
                if formatted:
                    item = {"text": formatted}
                    # 如果使用CoT-Valve++，添加beta值
                    if strategy == "valve_plus_plus" and chosen_index in beta_values:
                        item["beta"] = beta_values[chosen_index]
                    processed_examples.append(item)
    
    # 对数据集进行打包
    if len(processed_examples) == 0:
        raise ValueError("没有有效的训练样本！请检查数据集和处理逻辑。")
    
    return DatasetWrapper(processed_examples)

def get_valve_plus_p_dataset(dataset, current_stage, total_stages):
    """
    为CoT-Valve+P策略创建数据集。
    根据当前训练阶段，选择性地只使用特定长度范围的solution。
    
    Args:
        dataset: 原始数据集
        current_stage: 当前训练阶段，从0开始
        total_stages: 总训练阶段数
    
    Returns:
        当前阶段应该使用的数据子集
    """
    # 预处理数据集
    all_examples = preprocess_dataset_for_sft(dataset, use_all_solutions=True)
    
    # 按长度排序所有样本，从长到短
    examples_with_length = [(i, calculate_token_length(ex["text"])) for i, ex in enumerate(all_examples.examples)]
    sorted_indices = sorted(examples_with_length, key=lambda x: x[1], reverse=True)
    
    # 将样本分成total_stages个桶
    bucket_size = len(sorted_indices) // total_stages
    if bucket_size == 0:
        bucket_size = 1  # 确保至少每个桶有一个样本
    
    # 获取当前阶段对应的样本索引
    start_idx = current_stage * bucket_size
    end_idx = min(start_idx + bucket_size, len(sorted_indices))
    current_indices = [sorted_indices[i][0] for i in range(start_idx, end_idx)]
    
    # 创建当前阶段的数据子集
    current_examples = [all_examples.examples[i] for i in current_indices]
    
    print(f"CoT-Valve+P阶段{current_stage+1}/{total_stages}: 使用{len(current_examples)}个样本")
    
    return DatasetWrapper(current_examples)

def load_and_prepare_data(model_path=None, dataset_path=None, for_training=True, strategy=None, stage=None):
    """
    加载并准备数据。
    
    Args:
        model_path: 模型路径，用于加载tokenizer
        dataset_path: 数据集路径
        for_training: 是否用于训练
        strategy: 训练策略，可选值有"valve_plus_plus"和"valve_plus_p"
        stage: 如果使用CoT-Valve+P策略，当前是第几阶段（从0开始）
    
    Returns:
        处理后的数据集
    """
    # 确保tokenizer已加载
    get_tokenizer(model_path)
    
    # 确定数据集路径
    if dataset_path is None:
        dataset_path = DATA_CONFIG["train_dataset_path"] if for_training else DATA_CONFIG["eval_dataset_path"]
    
    config_name = DATA_CONFIG.get("dataset_config_name", "default")  # 使用default作为默认配置
    split = "train" if for_training else "test"  # 或"validation"
    
    print(f"Loading dataset from {dataset_path}, split={split}, config={config_name}")
    raw_dataset = load_dataset(dataset_path, name=config_name, split=split)
    
    if for_training:
        if strategy == "valve_plus_p" and stage is not None:
            # 渐进式训练的特定阶段
            total_stages = 5  # 假设总共5个阶段，可以根据需要调整
            return get_valve_plus_p_dataset(raw_dataset, stage, total_stages)
        else:
            # 标准SFT或CoT-Valve++
            return preprocess_dataset_for_sft(
                raw_dataset, 
                use_all_solutions=True, 
                strategy=strategy
            )
    else:
        # 评估数据集处理
        def format_eval(example):
            return {"text": format_prompt(example, solution_index=None)}
        
        processed_dataset = raw_dataset.map(format_eval, remove_columns=raw_dataset.column_names)
        return processed_dataset