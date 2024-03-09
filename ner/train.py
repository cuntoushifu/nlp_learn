import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
import json
import os
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, label_map, max_len=512):
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len
        self.data = []

        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                text, label_entities = entry['text'], entry['label']

                encoded_text = tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                                                     padding='max_length', truncation=True, return_offsets_mapping=True)
                input_ids, attention_mask, offsets = encoded_text['input_ids'], encoded_text['attention_mask'], \
                encoded_text['offset_mapping']

                labels = [self.label_map['O']] * len(input_ids)

                for label in label_entities:
                    start_char, end_char, label_type = label
                    label_id = self.label_map[label_type]
                    for idx, (start, end) in enumerate(offsets):
                        if start_char <= start < end_char or start_char < end <= end_char or (
                                start <= start_char and end >= end_char):
                            labels[idx] = label_id

                self.data.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.data[idx].items()}
        return item


label_map = {
    "提取货样（文本）": 0,
    "O": 1,
    "申报日期（文本）": 2,
    "贸易信息": 3,
    "申报媒介": 4,
    "主体": 5,
    "税率、汇率（文本）": 6,
    "申报方式形式（纸质、电子、联网等）": 7,
    "单证及说明文件": 8,
    "惩罚": 9,
    "补充申报": 10,
    "提前（文本）": 11,
    "特殊情况": 12,
    "集中申报": 13,
    "人员": 14,
    "地点（海关）": 15,
    "法规": 16,
    "许可证有效期": 17,
    "货物（空、进、出、进出口）": 18,
    "手续": 19,
    "逾期（文本）": 20,
    "时间节点": 21,
    "特殊运输方式（具体）": 22
}

tokenizer = BertTokenizerFast.from_pretrained("/workai/models/bert-base-chinese")

model = BertForTokenClassification.from_pretrained("/workai/models/bert-base-chinese", num_labels=len(label_map))

data_path = "/workai/yangye/ner/data/data.jsonl"
dataset = CustomDataset(data_path, tokenizer, label_map)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 示例：训练过程和模型保存逻辑需要根据具体需求实现

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数
loss_fn = torch.nn.CrossEntropyLoss()

# 训练参数
epochs = 32

# 训练循环
model.train()
for epoch in range(epochs):
    total_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Epoch {:1d}".format(epoch+1))
    for step, batch in progress_bar:
        # 将数据移动到设备上
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 清除先前的梯度
        model.zero_grad()

        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]

        # 后向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': '{:.3f}'.format(total_loss/(step+1))})

# 保存模型
model_save_path = 'model'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# 保存模型的权重和分词器
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("Training completed. Model saved to {}".format(model_save_path))