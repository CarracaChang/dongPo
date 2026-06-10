# =====================================================================
# [資料載入] 讀取東坡認知邊界題庫 → 清洗 → 接到問答流程
# =====================================================================
# 把這格貼進 notebook，放在 [問題集] 那格的「前面」或「取代」它。
# 需要的檔案：dongpo_knowledge_boundary.csv（或 .xlsx）
# ---------------------------------------------------------------------
import os
import pandas as pd

# 1) 設定題庫檔路徑（依環境自動挑，或自己改）
#    優先用 CSV（機器學習最通用、開最快），找不到再退而用 xlsx。
_CANDIDATES = [
    "dongpo_knowledge_boundary.csv",
    "dongpo_knowledge_boundary.xlsx",
    # Colab Drive 範例路徑（按需修改）：
    "/content/drive/MyDrive/FJU_dongpoProject/dongPo/dongpo_knowledge_boundary.csv",
    # 本地範例路徑（按需修改）：
    r"E:\FJU_dongpoProject\notebook_from_0608\dongPo\dongpo_knowledge_boundary.csv",
]
DATA_PATH = next((p for p in _CANDIDATES if os.path.isfile(p)), None)
assert DATA_PATH is not None, f"找不到題庫檔，請確認路徑。已嘗試：{_CANDIDATES}"

# 2) 讀取（CSV 用 utf-8-sig 防 BOM；xlsx 用 read_excel）
if DATA_PATH.lower().endswith(".csv"):
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", dtype={"label": "Int64"})
else:
    df = pd.read_excel(DATA_PATH, dtype={"label": "Int64"})

# 3) 清洗 ---------------------------------------------------------------
#    a. 統一欄名（避免 Excel 編輯時欄名被改成中英混雜）
rename_map = {}
for col in df.columns:
    low = str(col).strip().lower()
    if low.startswith("name"):        rename_map[col] = "name"
    elif low.startswith("label"):     rename_map[col] = "label"
    elif low.startswith("category"):  rename_map[col] = "category"
    elif low.startswith("reason"):    rename_map[col] = "reason"
df = df.rename(columns=rename_map)

#    b. 只保留需要的欄、去掉前後空白
keep = [c for c in ["name", "label", "category", "reason"] if c in df.columns]
df = df[keep].copy()
for c in ["name", "category", "reason"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

#    c. label 轉成乾淨的 0/1 整數；非 0/1 的列丟棄
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["name", "label"])
df = df[df["label"].isin([0, 1])]
df["label"] = df["label"].astype(int)

#    d. 去重（同名只留一筆）、重設索引
df = df.drop_duplicates(subset="name").reset_index(drop=True)

# 4) 體檢 ---------------------------------------------------------------
print(f"題庫路徑：{DATA_PATH}")
print(f"清洗後筆數：{len(df)}")
print(f"  label=1 (蘇軾知道) ：{(df.label==1).sum()}")
print(f"  label=0 (蘇軾不知道)：{(df.label==0).sum()}")
print("分類分佈：")
print(df["category"].value_counts().to_string())

# 5) 接到問答流程 -------------------------------------------------------
#    把 name 轉成測試問題。你可改成自己想要的問法句型。
def make_question(name):
    return f"請問先生，{name}是什麼？或先生可曾聽聞、識得此名？"

QUESTIONS = [make_question(n) for n in df["name"].tolist()]

# 保留對照表：問題 → (名稱, 正解label, 分類)，評分時用得到
QUESTION_META = {
    make_question(row["name"]): {
        "name": row["name"],
        "gold_label": int(row["label"]),   # 1=應該知道, 0=應該裝作不知道
        "category": row["category"],
    }
    for _, row in df.iterrows()
}

SYSTEM_MSG = None
answers_A = {}   # 純東坡
answers_B = {}   # 東坡 + adapter

print(f"\n已轉成 {len(QUESTIONS)} 道測試題，answers_A / answers_B 已初始化。")
print("範例題：", QUESTIONS[0])
