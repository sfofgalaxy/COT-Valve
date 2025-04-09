from datasets import load_dataset
'''
每个数据集的结构如下：

question: 问题文本。
answer: 原始答案。
target: 提取的最终答案。
solution_{0,1,2,3,4}: 不同长度的推理链。
solution_{1,2,3,4}_token: 每个推理链的 token 数量。
'''
class MixChainDataLoader:
    def __init__(self, dataset_name):
        # dataset_gsm8k = load_dataset("horseee/MixChain-Z-GSM8K")
        # dataset_prm12k = load_dataset("horseee/MixChain-Z-PRM12K")
        # dataset_limo = load_dataset("horseee/MixChain-C-LIMO")
        self.dataset = load_dataset(dataset_name)

    def load_data(self):
        return self.dataset
        
