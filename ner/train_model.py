import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
import json
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import numpy as np
import os


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
                                                     padding='max_length', truncation=True, return_offsets_mapping=True,
                                                     is_split_into_words=False)
                input_ids, attention_mask, offsets = encoded_text['input_ids'], encoded_text['attention_mask'], \
                encoded_text['offset_mapping']

                labels = [self.label_map['O']] * len(input_ids)

                for start_char, end_char, label_type in label_entities:
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


# 标签映射
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

data_path = "data/data.jsonl"
dataset = CustomDataset(data_path, tokenizer, label_map)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练逻辑
optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 16
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Epoch {:1d}".format(epoch + 1))
    for step, batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': '{:.3f}'.format(total_loss / (step + 1))})


# 评估逻辑
def evaluate(model, dataloader):
    model.eval()
    predictions, true_labels = [], []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

    # Flatten the lists
    predictions = [item for sublist in predictions for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]

    precision = precision_score(true_labels, predictions, average='macro', labels=np.unique(predictions),
                                zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', labels=np.unique(predictions), zero_division=0)
    f1 = f1_score(true_labels, predictions, average='macro', labels=np.unique(predictions), zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


eval_metrics = evaluate(model, val_loader)
print(f"Precision: {eval_metrics['precision']}")
print(f"Recall: {eval_metrics['recall']}")
print(f"F1 Score: {eval_metrics['f1']}")

# 保存模型
model_save_path = 'model'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("Training completed. Model saved to {}".format(model_save_path))
