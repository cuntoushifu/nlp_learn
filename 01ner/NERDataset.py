from torch.utils.data import Dataset
import torch

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label_map, max_token_len=128):
        self.tokenizer = tokenizer
        self.data = data
        self.label_map = label_map
        self.max_token_len = max_token_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # 获取文本和标签
        text = self.data[index]["text"]
        labels = self.data[index]["labels"]
        
        # 编码文本
        encoding = self.tokenizer(text,
                                  max_length=self.max_token_len,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors="pt")
        
        # 编码标签
        label_ids = [self.label_map[label] for label in labels]
        # 需要对标签进行padding以匹配输入的长度
        label_ids += [self.label_map["O"]] * (self.max_token_len - len(label_ids))
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

# 假设`dataset`已经加载，并且`label_map`已经准备好
# 创建数据集实例
# train_dataset = NERDataset(dataset["train"], tokenizer, label_map)
# test_dataset = NERDataset(dataset["test"], tokenizer, label_map)
