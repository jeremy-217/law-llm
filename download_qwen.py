from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "Qwen/Qwen1.5-7B-Chat"

    print(" 正在載入 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="./qwen_model"
    )

    print(" 正在載入模型（可能需要幾分鐘）...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="./qwen_model",
        device_map="auto",
        load_in_4bit=True  # 省記憶體（需有 bitsandbytes）
    )

    print(" 模型與 tokenizer 下載完成！")

    # 做一次推論測試
    prompt = "請判斷下列事實涉及哪些法律後果：甲酒駕撞人後逃逸。"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(" 正在進行推論...")
    outputs = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n=== 推論結果 ===\n")
    print(result)

if __name__ == "__main__":
    main()
