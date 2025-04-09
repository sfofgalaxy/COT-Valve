'''
微调设置
使用 Hugging Face 的 transformers 库加载你的模型，并进行微调。

加载模型
模型路径如下：

/data_sde/lyl/QwQ-32B-AWQ
/data_sde/lyl/deepseek-r1-distill-qwen-32b-awq
/data_sde/lyl/deepseek-r1-distill-qwen-14b-awq
加载模型和 tokenizer
'''

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from dataloader import MixChainDataLoader

training_args = TrainingArguments(
    output_dir="./results",                # 保存模型的路径
    evaluation_strategy="epoch",          # 每个 epoch 进行评估
    learning_rate=5e-5,                   # 学习率
    per_device_train_batch_size=4,        # 每个设备的训练 batch size
    per_device_eval_batch_size=4,         # 每个设备的评估 batch size
    num_train_epochs=3,                   # 训练轮数
    weight_decay=0.01,                    # 权重衰减
    save_strategy="epoch",                # 每个 epoch 保存模型
    logging_dir="./logs",                 # 日志路径
    logging_steps=50,                     # 日志记录频率
    fp16=True,                            # 使用混合精度训练
    gradient_accumulation_steps=8,        # 梯度累积
    max_steps=10000                       # 最大训练步数
)

# 加载模型和 tokenizer
model_paths = [
    "/data_sde/lyl/QwQ-32B-AWQ",
    "/data_sde/lyl/deepseek-r1-distill-qwen-32b-awq",
    "/data_sde/lyl/deepseek-r1-distill-qwen-14b-awq"
]

models = [AutoModelForCausalLM.from_pretrained(path) for path in model_paths]
tokenizers = [AutoTokenizer.from_pretrained(path) for path in model_paths]

def preprocess_function(examples, tokenizer, solution_index):
    """
    将问题和推理链转换为模型输入格式。
    solution_index: 使用哪个推理链（0, 1, 2, 3, 4）
    """
    inputs = examples["question"]
    targets = examples[f"solution_{solution_index}"]
    
    # 拼接问题和推理链
    model_inputs = tokenizer(inputs, text_target=targets, max_length=2048, truncation=True)
    return model_inputs

# 选择使用的推理链（例如 solution_1）
solution_index = 1
tokenizer = tokenizers[0]  # 使用第一个模型的 tokenizer

dataset = MixChainDataLoader("horseee/MixChain-Z-GSM8K")
processed_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer, solution_index), batched=True)

trainer = Trainer(
    model=models[0],                       # 使用第一个模型
    args=training_args,
    train_dataset=processed_dataset["train"],  # 训练集
    eval_dataset=processed_dataset["validation"],  # 验证集
    tokenizer=tokenizer
)

# 推理链长度控制。在推理时，可以通过调整推理链的长度（选择不同的 solution_index）来控制生成结果。
def generate_answer(question, model, tokenizer, solution_index):
    """
    使用微调后的模型生成答案。
    """
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 示例问题
question = "Amanda had 10 notebooks. This week, she ordered 6 more and then lost 2. How many notebooks does Amanda have now?"

# 使用微调后的模型生成答案
answer = generate_answer(question, models[0], tokenizers[0], solution_index=1)
print(answer)
# trainer.evaluate()
