"""用 unsloth + LoRA(4-bit) 微调 llama3.1:8b，得到论文中的 formal_spec_ft（论文 2.2.1 / 4）。

需要 NVIDIA GPU，并安装：pip install unsloth trl transformers datasets
超参数按论文：lr=2e-4，3 epochs，4-bit 量化。LoRA rank、batch size 等论文未给出，下面是常用默认值。
注意：trl 的 SFTTrainer/SFTConfig 参数在不同版本间有变动，如报错请对照你安装版本的文档。

训练完成后导出 GGUF，并注册到 ollama：
  ollama create formal_spec_ft -f outputs/formal_spec_ft/Modelfile
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

import config
from task_data import load_task_data
from tasks import FORMAL_TASK, build_messages

OUT = Path("outputs/formal_spec_ft")


def main(data_dir: str = "data"):
    train, _ = load_task_data("formal", data_dir)  # 只用训练集(50%)，测试集不参与
    print("train size:", len(train))

    model, tok = FastLanguageModel.from_pretrained(
        model_name=config.FT_BASE_MODEL,
        max_seq_length=config.FT_MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.FT_LORA_R,
        lora_alpha=config.FT_LORA_R,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=config.SEED,
    )

    # 训练样本与推理时 prompt（零示例）保持同一格式：system(模板) + user(意图) -> assistant(JSON)
    def to_text(it):
        msgs = build_messages(FORMAL_TASK, it.intent, [])
        msgs.append({"role": "assistant", "content": json.dumps(it.output, ensure_ascii=False)})
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    ds = Dataset.from_list([to_text(x) for x in train])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=config.FT_MAX_SEQ_LEN,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=config.FT_EPOCHS,
            learning_rate=config.FT_LR,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            warmup_steps=5,
            logging_steps=10,
            seed=config.SEED,
            output_dir=str(OUT / "ckpt"),
            report_to="none",
        ),
    )
    trainer.train()

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(str(OUT / "gguf"), tok, quantization_method="q4_k_m")
    # 生成 ollama Modelfile（GGUF 文件名以实际导出为准，必要时修改 FROM 行）
    gguf = next((OUT / "gguf").glob("*.gguf"), None)
    (OUT / "Modelfile").write_text(
        f"FROM {gguf.resolve() if gguf else './gguf/model.gguf'}\n"
        "PARAMETER temperature 0\n"
        f"PARAMETER num_ctx {config.NUM_CTX}\n",
        encoding="utf-8",
    )
    print("Done. Next: ollama create formal_spec_ft -f", OUT / "Modelfile")


if __name__ == "__main__":
    main()
