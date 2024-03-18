import json
import torch

def print_pretty_json(inputs):
    """
    接收一个包含torch.Tensor对象的字典，将其转换为列表，并打印美观的JSON字符串。

    参数:
    - inputs: 一个字典，其值包含torch.Tensor对象。
    """
    # 将torch.Tensor转换为列表
    inputs_for_json = {key: value.numpy().tolist() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

    # 使用json.dumps美化输出，并打印结果
    pretty_json = json.dumps(inputs_for_json, indent=4, ensure_ascii=False)
    print(pretty_json)
