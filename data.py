"""从 Hugging Face 加载 NetConfEval Formal Specification Translation 数据。

逻辑：
1. 从 NetConfEval/NetConfEval 下载数据
2. 过滤复杂度小于 27 的样本
3. 不再加载 NFV 论文用例
4. 按 0.5:0.5 划分 train/test
"""

from __future__ import annotations

from datasets import load_dataset
from sklearn.model_selection import train_test_split

HF_DATASET_PATH = "NetConfEval/NetConfEval"
HF_DATASET_CONFIG = "Formal Specification Translation"
MAX_COMPLEXITY = 10
TRAIN_RATIO = 0.5
SEED = 42


def load_netconfeval_formal_data():
    """读取 HF 数据集并返回过滤后的 train/test 列表。"""
    print("正在加载 NetConfEval 数据集...")
    
    dataset = load_dataset(HF_DATASET_PATH, HF_DATASET_CONFIG)
    all_data = dataset["train"]

    filtered_data = [
        item for item in all_data
        if int(item.get("batch_size", 0)) < MAX_COMPLEXITY
    ]

    print(f"过滤后剩余 {len(filtered_data)} 条数据。")

    # 论文要求按 0.5:0.5 划分，固定随机种子
    train_data, test_data = train_test_split(
        filtered_data,
        train_size=TRAIN_RATIO,
        test_size=1.0 - TRAIN_RATIO,
        random_state=SEED,
        shuffle=True,
    )

    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = load_netconfeval_formal_data()
    sample = train_data[0]
    print("\n--- 示例数据 ---")
    print("输入提示 (Prompt/Intent):", sample.get("human_language", "N/A"))
    print("期望输出 (Expected JSON):", sample.get("expected", "N/A"))