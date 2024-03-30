# %%
from utils import *
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments

from transformers import DataCollatorForTokenClassification


# %%
# 指定文件路径
train_file_path = 'data/train.json'  # 训练集文件路径
test_file_path = 'data/test.json'    # 验证集（或测试集）文件路径

# 加载数据集
dataset = load_dataset('json', data_files={'train': train_file_path, 'test': test_file_path})


# %%
# 选择训练集中的前5个样本
sampled_data = dataset["train"].select(range(5))

# 打印每个样本
for i, sample in enumerate(sampled_data):
    print(f"Sample {i}: {sample}\n")


# %%
label_map,label_nums=extract_label_info(dataset)
label_map,label_nums

# %%

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=label_nums)


# %%
# 假设每个样本的标签都存储在"labels"字段中，并且文本存储在"text"字段中
def encode_labels(example):
    # 将文本标签转换为数字ID
    return {"labels": [label_map[label] for label in example["labels"]]}


# %%
dataset = dataset.map(encode_labels)

# %%
# 确保你的encode_labels和tokenize_and_encode函数适应了BertTokenizerFast
def tokenize_and_encode(examples):
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128, return_offsets_mapping=True)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            # 检查word_idx是否有效
            if word_idx is not None and word_idx < len(label):
                label_ids.append(label_map.get(label[word_idx], -100))
            else:
                label_ids.append(-100)  # 对于无法映射的令牌，使用-100
        labels.append(label_ids)

    
    tokenized_inputs["labels"] = labels
    tokenized_inputs.pop("offset_mapping") # 在返回之前移除offset_mapping
    return tokenized_inputs

# 使用map函数应用编码函数，记得使用batched=True来批量处理
tokenized_datasets = dataset.map(tokenize_and_encode, batched=True, remove_columns=["text", "labels"])




# %%

# 创建DataCollator实例
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="pt")

training_args = TrainingArguments(
  "my-bert-chinese",
  num_train_epochs=3,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=32,
  warmup_steps=100,
  weight_decay=0.01,
  logging_strategy="epoch",  # 设置日志记录策略为每个epoch
  evaluation_strategy="epoch",  # 设置评估策略为每个epoch，如果需要评估的话
  save_strategy="epoch",  # 设置保存策略为每个epoch，如果你也想按epoch保存模型的话
  load_best_model_at_end=True,  # 训练结束时加载最佳模型
 )

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator
)
trainer.train()

# %%
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
print(type(tokenizer))



