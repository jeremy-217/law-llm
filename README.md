# Qwen Legal Domain Fine-tuning with LoRA

本專案使用 Qwen 模型結合 LoRA（Low-Rank Adaptation）技術，針對繁體中文法律領域資料進行微調。目的是建立一個能夠理解並回答法律問題的輕量化語言模型。

## 專案結構

- `download_qwen.py`：下載 Hugging Face 上的 Qwen 模型（例如 Qwen-7B 或 Qwen-1.5 系列）
- `legal_data.jsonl`：訓練用的資料集，格式為 JSON Lines，每行一筆資料
- `train_lora_legal.py`：使用 Hugging Face Transformers 與 PEFT（LoRA）進行微調的訓練腳本

## 安裝套件

請使用 Python 3.8 或以上版本，並安裝以下相依套件：
pip install transformers peft datasets accelerate bitsandbytes

如果你的硬體支援 flash-attn，可選擇加速訓練與推論：
pip install flash-attn

資料格式說明
legal_data.jsonl 為 JSON Lines 格式，每行一筆資料，格式如下：

{"input": "民法第184條的規定是什麼？", "output": "民法第184條是關於侵權行為的條文，主要內容是..."}
請確保資料經過適當清理、編碼為 UTF-8，並避免重複與錯誤資訊。

使用步驟
第一步：下載 Qwen 模型
python download_qwen.py
此步驟會將 Hugging Face 上的模型下載至本地（模型名稱與儲存路徑可在腳本中修改）。

第二步：進行模型微調
python train_lora_legal.py \
  --base_model_path ./qwen_model \
  --train_data_path ./legal_data.jsonl \
  --output_dir ./qwen_lora_legal
如需調整訓練參數（例如學習率、LoRA rank、batch size、epochs 等），可直接在 train_lora_legal.py 檔案內修改。

模型推論範例
微調完成後，可使用以下程式載入模型與進行推論：
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("Qwen")
base_model = AutoModelForCausalLM.from_pretrained("Qwen")
model = PeftModel.from_pretrained(base_model, "./qwen_lora_legal")

inputs = tokenizer("請解釋定作契約的要件", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

注意事項
建議使用具備 16GB 或以上 VRAM 的 GPU 進行訓練。

本專案所訓練之模型僅供實驗用途，不得用於實際法律判斷或提供法律建議。

若使用真實資料進行訓練，請自行確認是否涉及個資或智慧財產權等問題。
