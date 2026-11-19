"""全局配置：对应论文 Table 1 / Section 4 的实验设置。"""

# 论文使用的 6 个离线 LLM（ollama 标签）。先执行 `ollama pull <tag>`
MODELS = [
    "qwen2:7b",
    "llama3.1:8b",
    "gemma2:9b",
    "gemma2:27b",
    "qwen2:72b",
    "llama3.1:70b",
]
# 微调后的模型（基于 llama3.1:8b，见 finetune.py），用 `ollama create` 注册后使用
FINETUNED_MODEL = "formal_spec_ft"

# 文本嵌入模型（论文未指明具体型号，这里用 ollama 上常用的 nomic-embed-text）
EMBED_MODEL = "nomic-embed-text"

# 论文 4.1：in-context 示例数 M。70B/72B 因输入长、超时，只用 1 个
DEFAULT_M = 9
M_FOR_BIG_MODELS = {"qwen2:72b": 1, "llama3.1:70b": 1}

# 推理参数
TEMPERATURE = 0.0
NUM_CTX = 8192          # 上下文长度，示例多时可以调大
MAX_RETRIES = 3         # 系统验证失败后的最大重试次数（论文 3.3 的 maximum retry limit）

# 数据集
FORMAL_MAX_COMPLEXITY = 27   # 只用复杂度指数 < 27 的 NetConfEval 子集
FORMAL_TRAIN_RATIO = 0.5     # 0.5:0.5 划分
SEED = 42

# LoRA 微调（论文 4：lr=0.0002, 3 epochs, 4-bit 量化, unsloth）
FT_BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
FT_LR = 2e-4
FT_EPOCHS = 3
FT_LORA_R = 16
FT_MAX_SEQ_LEN = 8192