import torch
from transformers import BertTokenizerFast, BertForTokenClassification


def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForTokenClassification.from_pretrained(model_path)
    return model, tokenizer


def predict_entities(text, model, tokenizer, label_map):
    id_to_label = {id: label for label, id in label_map.items()}
    inputs = tokenizer.encode_plus(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length',
                                   return_offsets_mapping=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    offset_mapping = inputs["offset_mapping"].squeeze().tolist()

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    current_entity = ""
    current_label = None
    entities = []

    for idx, (offset, prediction) in enumerate(zip(offset_mapping, predictions)):
        if idx == 0 or idx == len(predictions) - 1:  # Ignore [CLS] and [SEP]
            continue
        token_label = id_to_label[prediction]
        start, end = offset
        if token_label != "O":
            if token_label == current_label:
                current_entity += text[start:end]
            else:
                if current_entity:
                    entities.append(f"{current_entity} ({current_label})")
                current_entity = text[start:end]
                current_label = token_label
        else:
            if current_entity:
                entities.append(f"{current_entity} ({current_label})")
                current_entity = ""
                current_label = None

    if current_entity:  # Catch any final entity
        entities.append(f"{current_entity} ({current_label})")

    return entities


if __name__ == "__main__":
    model_path = 'model'
    model, tokenizer = load_model_and_tokenizer(model_path)
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
    print("Type 'quit' to exit.")
    while True:
        text = input("Enter text: ")
        if text.lower() == "quit":
            break
        entities = predict_entities(text, model, tokenizer, label_map)
        print("Identified Entities:")
        for entity in entities:
            print(entity)
