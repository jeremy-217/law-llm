from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

# 模型名稱（小型中文模型）
model_name = "uer/gpt2-chinese-cluecorpusswwm"
data_path = "legal_data.jsonl"  # 你的資料檔案

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 載入資料集
dataset = load_dataset("json", data_files=data_path)["train"]

# 格式化資料（將 instruction + input 合併成 prompt）
def format(example):
    prompt = f"### 指令:\n{example['instruction']}\n\n### 事實:\n{example['input']}\n\n### 回覆:\n{example['output']}"
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(format)

# 載入模型（不使用 4bit，不使用 trust_remote_code）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"}  # 或改成 "auto" 看你的設備是否支援 MPS
)

# 設定 LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],  # GPT2 的 Attention 線性層名稱
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# 訓練參數
training_args = TrainingArguments(
    output_dir="./gpt2-lora-legal",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    learning_rate=2e-4,
    save_total_limit=2
    # 不要使用 fp16 或 bf16，macOS MPS 不支援
)

# 建立 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# 開始訓練
trainer.train()

# 儲存訓練後的 LoRA 權重
model.save_pretrained("./gpt2-lora-legal")
