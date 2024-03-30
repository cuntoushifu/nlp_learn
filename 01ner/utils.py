import json
from collections import OrderedDict  # 导入OrderedDict

def preprocess_and_save_data(input_file_path, output_file_path):
    """
    预处理输入文件中的数据并保存到输出文件中。

    该函数首先从指定的输入文件路径读取数据，然后处理这些数据，最后将处理后的数据保存到指定的输出文件路径。
    数据处理的主要目的是将文本分割成句子和对应的标签列表，并以JSON格式保存。

    参数:
    - input_file_path: 输入文件的路径，包含待处理的原始文本数据。
    - output_file_path: 输出文件的路径，处理后的数据将保存为JSON格式。
    """

    # 定义内部函数load_and_process_data来加载和处理数据
    def load_and_process_data(file_path):
        """
        加载和处理指定路径的文件数据。

        该函数读取文件，每一行代表一个词及其标签。空行表示句子的分割。
        函数返回处理后的句子列表和相应的标签列表。

        参数:
        - file_path: 要处理的文件路径。

        返回:
        - sentences: 处理后的句子列表，每个句子是一个字符串。
        - labels: 对应于句子的标签列表，每个标签列表包含与句子中每个词对应的标签。
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 初始化句子和标签列表
        sentences = []
        labels = []

        # 初始化当前处理的句子和标签
        sentence = []
        label = []
        for line in lines:
            line = line.strip()
            if not line:  # 空行表示句子的结束
                if sentence and label:
                    # 将当前句子和标签添加到列表中，并重置它们
                    sentences.append(''.join(sentence))
                    labels.append(label)
                    sentence = []
                    label = []
                continue
            word, tag = line.split()  # 分割每一行为词和标签
            sentence.append(word)  # 添加词到当前句子
            label.append(tag)  # 添加标签到当前标签列表

        # 添加最后处理的句子和标签
        if sentence and label:
            sentences.append(''.join(sentence))
            labels.append(label)

        return sentences, labels

    # 调用内部函数处理输入文件的数据
    sentences, labels = load_and_process_data(input_file_path)

    # 准备要保存的数据
    data_to_save = []
    for sent, lab in zip(sentences, labels):
        data_to_save.append({'text': sent, 'labels': lab})

    # 以JSON格式保存处理后的数据
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    print(f"Data has been processed and saved to {output_file_path}")


# # 指定输入文件路径和输出文件路径
# input_file_path = '01ner/data/train.txt'  # 替换为你的输入文件路径
# output_file_path = '01ner/data/train.json'  # 替换为你想要的输出文件路径

# # 调用函数进行数据处理和保存
# preprocess_and_save_data(input_file_path, output_file_path)


import json

def extract_entity_types_from_json(json_file_path, output_txt_path):
    """
    从指定的JSON文件中读取NER数据集，提取所有独特的实体类型，并将这些实体类型保存到TXT文件中。
    
    参数:
    - json_file_path: str, 输入的JSON文件路径，该文件包含NER标注数据。
    - output_txt_path: str, 输出的TXT文件路径，用于保存提取出的独特实体类型。
    
    用法示例:
    json_file_path = 'path/to/your/data.json'
    output_txt_path = 'path/to/your/entities.txt'
    extract_entity_types_from_json(json_file_path, output_txt_path)
    """

    # 存储唯一的实体类型
    unique_entities = set()

    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历JSON文件中的每个条目
    for item in data:
        # 假设每个条目中包含'labels'字段，其中包含了标签列表
        labels = item['labels']
        # 遍历每个标签，提取实体类型
        for label in labels:
            # 只处理以'B-'或'I-'开头的标签，这些标签表示实体的开始和内部
            if label.startswith('B-') or label.startswith('I-'):
                # 从标签中提取出实体类型，并添加到集合中以去重
                entity_type = label.split('-', 1)[1]
                unique_entities.add(entity_type)

    # 将所有独特的实体类型写入到TXT文件中，每个实体类型占一行
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for entity in sorted(unique_entities):  # 对实体类型进行排序，以便于阅读
            f.write(entity + '\n')

    print(f"Extracted entity types have been written to {output_txt_path}")

# 示例用法
# 注意替换下面的路径为你的实际文件路径
# json_file_path = '01ner/data/train.json'
# output_txt_path = './01ner/data/labels.txt'
# extract_entity_types_from_json(json_file_path, output_txt_path)

def extract_label_info(datasets):
    """
    从datasets对象中提取label_map和label_nums。

    参数:
    - datasets: 一个datasets对象，其中包含NER标签。

    返回:
    - label_map: 一个从标签字符串到唯一索引的映射。
    - label_nums: 标签的总数。
    """
    unique_labels = set()
    for split in datasets:
        for example in datasets[split]:
            labels = example['labels']
            unique_labels.update(labels)

    # 创建label_map，索引从1开始，预留0给"O"标签
    label_map = {label: idx + 1 for idx, label in enumerate(sorted(unique_labels))}
    # 将"O"标签添加到label_map，假设"O"标签表示非实体部分
    label_map["O"] = 0

    # 计算label_nums
    label_nums = len(label_map)

    return label_map, label_nums



def generate_label_map_and_count(json_file_path):
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 提取所有独特的标签
    unique_labels = set(label for sample in data for label in sample["labels"])
    
    # 创建从标签到索引的映射（确保'O'标签是0）
    label_map = OrderedDict({"O": 0})
    for label in sorted(unique_labels):
        if label != "O":
            label_map[label] = len(label_map)
    
    # 计算标签的数量
    label_nums = len(label_map)
    
    return label_map, label_nums

# preprocess_and_save_data("./01ner/data/train.txt", "./01ner/data/train.json")、

def display_text_labels(text, predictions, labels, tokenizer):
    """
    根据模型预测和分词器显示文本及其标签。
    
    参数:
    text (str): 原始文本。
    predictions (list): 模型对每个令牌的预测标签索引。
    labels (dict): 索引到标签的映射。
    tokenizer: 使用的分词器实例。
    """
    # 将文本转换为模型可接受的格式（添加特殊令牌）
    encoded_input = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = encoded_input["input_ids"][0].tolist()  # Tensor转换为list
    tokens = tokenizer.convert_ids_to_tokens(input_ids)  # ID转换为令牌

    # 映射预测标签索引回标签
    predicted_labels = [labels[pred] for pred in predictions]

    print("令牌\t预测标签")
    print("------------------")
    for token, label in zip(tokens, predicted_labels):
        # 忽略特殊令牌的打印
        if token not in ["[CLS]", "[SEP]", "[PAD]"]:
            print(f"{token}\t{label}")


