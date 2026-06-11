# ============================================================
#   蘇東坡 AI 雙模組協作中轉站
#   Su Dongpo AI Dual-Module Collaboration Relay Station
# ============================================================

import sys
import torch
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# 一、全域參數設定（Global Variables）
# ============================================================
GEMINI_API_KEYS    = [
    "",  # 主力金鑰（帳號 A）
    ""   # 備用金鑰（帳號 B）
]                                                             # ← 請填入您的 Google Gemini API 金鑰
CURRENT_KEY_INDEX  = 0  # 紀錄目前正在使用哪一把鑰匙（0 代表第一把，1 代表第二把）
BASE_MODEL_ID      = "QingYuYunTu/DongPo"
LORA_WEIGHT_PATH   = r"C:\Users\gr104\LLaMA-Factory\saves\Qwen2.5-7B-Instruct\lora\500+150RAG"

# ============================================================
# 二、模組 1：系統初始化（Initialization）
# ============================================================

def initialize_gemini():
    """初始化 Google Gemini API 客戶端，回傳模型物件。"""
    genai.configure(api_key=GEMINI_API_KEYS[CURRENT_KEY_INDEX])
    
    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=(
            "你是一個專屬蘇東坡的歷史背景知識庫。請提供客觀、簡明的史實。"
            "【嚴格禁止】自稱人工智慧或AI。"
            "若使用者詢問蘇東坡的第一人稱問題（如：你喜歡吃什麼？），請直接輸出蘇東坡的歷史生平事實（如：蘇軾在黃州喜歡吃豬肉、竹筍等）。"
            "若使用者詢問後世歷史，請直接給出該歷史事件的客觀簡介。"
        ),
    )
    print("[初始化] ✔ Gemini API 客戶端設定完成。")
    return gemini_model


def initialize_local_model():
    """
    以 4-bit 量化載入 Qwen 2.5 7B 基底模型，
    再融合 LoRA 權重，回傳 (tokenizer, model)。
    """
    print(f"[初始化] 開始載入基底模型：{BASE_MODEL_ID}")
    print("[初始化] 啟用 4-bit 量化（BitsAndBytes NF4），節省 VRAM …")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,   # 雙量化，進一步降低顯存佔用
        bnb_4bit_quant_type="nf4",        # NF4 量化類型
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",               # 自動分配 GPU/CPU
        trust_remote_code=True,
    )
    print("[初始化] ✔ 基底模型載入完成。")

    print(f"[初始化] 開始融合 LoRA 權重：{LORA_WEIGHT_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_WEIGHT_PATH)
    model.eval()
    print("[初始化] ✔ LoRA 權重融合完成，模型就緒。")

    return tokenizer, model


# ============================================================
# 三、模組 2：事實查核中樞（Fact-Checking）
# ============================================================

def get_historical_facts(gemini_model, user_question: str) -> str:
    global CURRENT_KEY_INDEX, gemini_model_global # 如果有需要，確保能動態更新
    try:
        response = gemini_model.generate_content(user_question)
        return response.text.strip()
    except Exception as exc:
        # 萬一在台上遇到 429 流量超限，立刻自動換第二把免費鑰匙！
        if "429" in str(exc) and CURRENT_KEY_INDEX < len(GEMINI_API_KEYS) - 1:
            print("\n[系統] 觸發流量限制，正在自動切換至備用金鑰重試...")
            CURRENT_KEY_INDEX += 1
            genai.configure(api_key=GEMINI_API_KEYS[CURRENT_KEY_INDEX])
            # 使用新鑰匙重新建立一個模型物件並請求
            new_model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=gemini_model.config.system_instruction if hasattr(gemini_model, 'config') else None
            )
            try:
                response = new_model.generate_content(user_question)
                return response.text.strip()
            except Exception as e:
                return f"（備用金鑰亦查詢失敗：{e}）"
        
        print(f"[警告] Gemini API 呼叫失敗：{exc}")
        return "（史實查詢失敗，請確認 API 金鑰與網路連線是否正常）"


# ============================================================
# 四、模組 3：動態提示詞組裝（Dynamic Prompting）
# ============================================================

def assemble_prompt(historical_facts: str, user_question: str) -> str:
    """
    將史實與原始問題依照嚴格格式拼接為結構化提示詞。
    加入「時空認知隔離」指引，防止模型產生現代人視角的幻覺。
    """
    prompt = (
        "【史實參考】\n"
        f"{historical_facts}\n"
        "\n"
        "【處理指引】\n"
        "若上述史實發生在北宋（1101年）之後，請保持你『毫不知情』的設定。你可以假裝是從使用者的提問中初次聽聞此事，並用大宋的經驗進行類比、震驚或感慨，絕對不可表現出你原本就熟知明清或現代歷史的態度。\n"
        "\n"
        "【問題】\n"
        f"{user_question}\n"
    )
    return prompt


# ============================================================
# 五、模組 4：本地模型推論（Local Generation）
# ============================================================

_SUDONGPO_SYSTEM = (
    "你現在在是宋代文人蘇軾（蘇東坡），靈魂穿越至當代，記憶與價值觀停留在宋朝。"
    "一律使用繁體中文，融合文言與白話，展現豁達幽默的個性。"
    "對於宋代之後的事物，以時空旅人身份表達好奇或以宋代經驗類比，"
    "切勿直接以現代人視角回答。"
)


def generate_response(tokenizer, model, assembled_prompt: str) -> str:
    """
    將組裝好的提示詞套入 Qwen Chat Template，
    呼叫本地 LoRA 模型生成蘇東坡風格回答。
    """
    messages = [
        {"role": "system",  "content": _SUDONGPO_SYSTEM},
        {"role": "user",    "content": assembled_prompt},
    ]

    # 使用 Qwen 預設 Chat Template 編碼（add_generation_prompt 加入 <|im_start|>assistant）
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_length = model_inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 僅解碼新生成的 token，去除輸入部分
    output_ids = generated_ids[0][input_length:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return response.strip()


# ============================================================
# 六、程式進入點（Main Execution）
# ============================================================

def main():
    print("=" * 60)
    print("      蘇東坡 AI 雙模組協作中轉站  v1.0")
    print("=" * 60)

    # ── 初始化兩個模組 ──────────────────────────────────────
    gemini_model = initialize_gemini()
    tokenizer, local_model = initialize_local_model()

    print("\n[系統就緒] 輸入問題開始對話，輸入 'exit' 或 'quit' 結束程式。\n")

    # ── 主互動迴圈 ──────────────────────────────────────────
    while True:
        try:
            user_input = input("您的問題：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[系統] 程式已中止。")
            sys.exit(0)

        # 空白輸入直接跳過
        if not user_input:
            continue

        # 結束指令
        if user_input.lower() in ("exit", "quit"):
            print("[系統] 感謝使用，程式結束。")
            sys.exit(0)

        # Step 1 ── 事實查核
        print("\n[Step 1] 正在透過 Gemini 查詢客觀史實 …")
        facts = get_historical_facts(gemini_model, user_input)

        # Step 2 ── 提示詞組裝
        print("[Step 2] 組裝結構化提示詞 …")
        assembled = assemble_prompt(facts, user_input)

        # Step 3 ── 本地推論
        print("[Step 3] 本地 LoRA 模型推論中，請稍候 …\n")
        answer = generate_response(tokenizer, local_model, assembled)

        # ── 輸出結果 ────────────────────────────────────────
        print("=" * 60)
        print("【外部 API 擷取之史實】")
        print(facts)
        print("-" * 60)
        print("【蘇東坡 AI 最終回答】")
        print(answer)
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()
