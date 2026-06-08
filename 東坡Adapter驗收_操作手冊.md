# 東坡 LoRA Adapter 驗收 Notebook — 操作手冊

> 這份手冊教你怎麼用 `dongpo_adapter_verify_v2.ipynb` 這個檔案，把「純東坡」跟「東坡+adapter」兩個版本各問同一批問題，然後比較差異。
>
> 跟著步驟一步一步點，不需要懂程式。

---

## ⚠️ 先讀這段，讀完再開始（很重要）

**這個 Notebook 要跑 80 億參數（8B）的大模型，很吃顯示卡記憶體（VRAM）。**

- 本手冊**不保證**你的電腦顯卡有足夠的 VRAM。
- 本手冊**不保證**你的 Colab 帳號分配到的機器容量夠用。
- 跑得動跑不動、會不會卡住、會不會閃退，**取決於你自己的硬體**，這不在這份手冊的負責範圍內。

簡單講：**硬體不夠就是跑不動，這跟 Notebook 寫得好不好無關。** 請自行確認你的環境，再決定要不要跑。

一個粗略參考（不是保證）：
- 8B 模型在 4-bit 量化下，**載入後大約佔 5～6 GB VRAM**，生成時還要再多吃一些。
- VRAM 太少會發生：載入超級慢（好幾分鐘）、卡在某個百分比不動、或直接報錯。
- 看到載入速度是「每個分片要好幾秒」而不是「每秒好幾個」，通常就是 VRAM 不夠、資料被擠到一般記憶體去了。

---

## 你需要先準備好的東西

不管你跑哪一種，這兩樣一定要先有：

1. **東坡底模（Base Model）**：`QingYuYunTu/DongPo` 這個模型的完整檔案。
2. **Adapter**：要驗收的那包 LoRA，資料夾裡至少要有 `adapter_config.json` 和 `adapter_model.safetensors`。

---

## 你要先做一個選擇：跑 Colab 還是跑本地？

| | **Colab（雲端）** | **本地（自己的電腦）** |
|---|---|---|
| 適合誰 | 自己電腦顯卡不行、或懶得裝環境 | 有夠力的 NVIDIA 顯卡、想離線跑 |
| 要錢嗎 | 免費版有機會、但不保證分到好機器 | 不用，但要自己有顯卡 |
| 麻煩度 | 要把模型放到 Google Drive | 要自己裝 Python 環境 |

**這份 Notebook 會自動偵測你在哪個環境，自動切換路徑，你不用手動改程式。** 你只要把檔案放對位置就好。

---

# 路線 A：在 Google Colab 上跑

### A-1. 上傳 Notebook 到 Colab
1. 打開 [https://colab.research.google.com](https://colab.research.google.com)
2. 左上角「檔案 → 上傳筆記本」，把 `dongpo_adapter_verify_v2.ipynb` 丟進去。

### A-2. 開啟 GPU
1. 上方選單「執行階段 → 變更執行階段類型」。
2. 硬體加速器選 **T4 GPU**（或你有付費的話選更好的）。
3. 按儲存。

> ⚠️ 如果這裡選不到 GPU、或只給你 CPU，那就是你的帳號目前分不到顯卡。**這不是 Notebook 的問題，是你的額度問題。** 換時間再試，或考慮付費。

### A-3. 把模型放到 Google Drive
你有兩個做法，挑一個：

**做法一（推薦，最省事）：用 Notebook 幫你下載。**
- 找到標題寫 `[Colab下載] Colab下載模型腳本` 的那格（程式碼第 13 格附近）。
- 直接執行它。它會掛載你的 Drive、自動把東坡底模下載到
  `MyDrive/FJU_dongpoProject/dongPo/models/DongPo-Base`。
- 第一次下載約 15GB，要等幾分鐘，**保持分頁開著別關**。

**做法二：自己手動上傳。**
- 把東坡底模資料夾傳到 Drive 的 `MyDrive/FJU_dongpoProject/dongPo/models/DongPo-Base`。
- Adapter 傳到 `MyDrive/FJU_dongpoProject/dongPo/models/DongPo-Adapter`。

> 注意：Adapter 沒有自動下載腳本，**一定要自己手動放到 `DongPo-Adapter` 那個資料夾。**

### A-4. 開始照順序執行（重點來了）
從上往下，一格一格按「執行」（按左邊的播放鍵，或按 `Shift + Enter`）。

依序執行這些：
1. **裝套件那格**（第 2 格，開頭寫「Colab / 全新環境執行」）→ 執行。
2. **環境設定那格**（`import gc`，會偵測 GPU、設定量化）→ 執行。看到印出你的 GPU 名稱就對了。
3. **路徑設定那格**（第 15 格，`import os` 開頭）→ 執行。
   - 它會自動掛載 Drive、把模型從 Drive 複製到 Colab 本地高速碟（約 1～3 分鐘）。
   - 最後看到 **`✅ 路徑與必要檔案正確讀取完畢`** 才算成功。
   - 如果這裡報錯說找不到檔案，代表你 A-3 的模型沒放對位置，回去檢查。
4. **問答函式那格**（第 21 格，`import time` 開頭）→ 執行。看到印出「目前模式：採樣 sample」就對了。
5. **問題集那格**（`QUESTIONS = [...]`）→ 執行。

接著跳到下面「**共通：開始問答與比較**」那段。

---

# 路線 B：在自己的電腦（本地）跑

### B-1. 裝環境（只需做一次）
打開 Anaconda Prompt（或你的終端機），照順序貼上：

```bash
conda create -n dongpo python=3.11 -y
conda activate dongpo
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate bitsandbytes jupyter ipykernel
python -m ipykernel install --user --name dongpo --display-name "Python (dongpo)"
```

> 上面 `cu121` 是配 CUDA 12.1 的版本。你的顯卡驅動要支援。如果你不知道這是什麼，代表你的環境可能還沒準備好，請先弄懂再繼續。

### B-2. 把模型放到電腦裡
- 東坡底模放到：`E:\FJU_dongpoProject\notebook_from_0608\dongPo\models\DongPo-Base`
- Adapter 放到：`E:\FJU_dongpoProject\notebook_from_0608\dongPo\models\DongPo-Adapter`

> 這兩個路徑是 Notebook 裡寫死的。如果你要放別的地方，就要去第 15 格（路徑設定）把 `else:` 底下那兩行 `BASE_MODEL` / `ADAPTER_PATH` 改成你的路徑。

### B-3. 用對的環境開啟 Notebook
- 用 **VS Code**：打開 ipynb，右上角的 kernel 選 **`Python (dongpo)`**。
- 用 **Jupyter**：終端機打 `jupyter notebook`，開檔後 Kernel → Change Kernel → `Python (dongpo)`。

### B-4. 照順序執行
1. **降版那格**（第 3 格，寫「本地(3070)環境 降板」）→ **只有第一次需要執行**。
   - 它會把 transformers 鎖到 4.51.3 這個能認得東坡（Qwen3）的版本。
   - **執行完一定要重啟 kernel**（VS Code 右上 Restart），然後從頭再跑。
   - 之後再開就不用再跑這格了。
2. **不要執行第 2 格**（那是 Colab 用的 `%pip install -U`，本地跑會把你剛鎖好的版本又裝成最新的、然後壞掉）。
3. **環境設定那格**（`import gc`）→ 執行。看到印出 GPU 名稱就對了。
4. **路徑設定那格**（第 15 格）→ 執行，看到 `✅ 路徑與必要檔案正確讀取完畢`。
5. **問答函式那格**（第 21 格）→ 執行。
6. **問題集那格** → 執行。

接著看下面「共通」那段。

---

# 共通：開始問答與比較

> 不管你 Colab 還是本地，前面都跑完、看到路徑驗證成功之後，從這裡開始操作都一樣。

### 第 1 步：（可選）選擇解碼模式
在問答函式那格（第 21 格）最上面有一行開關：

```python
USING_GREEDY = False
```

- `False`＝**採樣模式**：東坡講話比較生動、有變化，但同樣問題每次答案可能不同。
- `True` ＝**貪婪模式（greedy）**：答案最穩定、最能重現，**適合公平比較 A 跟 B**。

改完這行，**一定要重新執行第 21 格**才會生效。
（之後存檔時，greedy 模式的檔名會自動加上 `greedy_` 前綴，方便你分辨。）

### 第 2 步：載入 A（純東坡）並問答
1. 執行 **`[A] 讀取純東坡`** 那格（第 29 格）。
   - 等它印出 `已載入 A：純東坡` 和顯存佔用數字。
   - 第一次載入會比較久，耐心等。
2. 執行 **`[A] 問答`** 那格（第 31 格）。它會把每一題的答案邊跑邊印出來，並存進記憶體。

### 第 3 步：換成 B（東坡+adapter）
**這步最容易出錯，看仔細：**

因為一張顯卡通常塞不下兩個模型，換模型前要先把 A 清掉。

- **最保險的做法（推薦）：直接重啟 kernel / 重啟執行階段。**
  - Colab：「執行階段 → 重新啟動執行階段」。
  - 本地：VS Code 右上「Restart」。
  - 重啟後，從前面的「環境設定 → 路徑 → 問答函式 → 問題集」**重新跑一遍**（這些很快），但 **A 問答那兩格不用再跑**。
- 或者，試試 **強化版 unload 那格**（第 27 格）清顯存。但對這種大模型不一定清得乾淨，清完顯存沒明顯下降的話，還是乖乖重啟。

> **重啟會不會弄丟 A 的答案？** 會。所以如果你用重啟大法，建議在重啟前先按存檔按鈕把 A 的結果存成檔案（見下面「存檔」）。

清乾淨之後：
1. 執行 **`[B] 讀取東坡+adapter`** 那格（第 33 格）。等它印出 `已載入 B：東坡 + adapter`。
2. 執行 **`[B] 問答`** 那格（第 35 格）。

### 第 4 步：看並排比較
執行比較那格（第 37 格），它會把每一題的「A 的答案」跟「B 的答案」並排印出來，讓你直接對照。

重點看三件事：
- B（套了 adapter）的東坡風格有沒有比 A 更濃？
- 史實有沒有答得更準？
- 「智慧型手機是什麼？」這種現代問題，B 有沒有「破功」去正常回答？（如果有，代表 adapter 可能污染了東坡「不知今事」的設定，這是壞消息。）

### 第 5 步：存檔
拉到最下面那格（第 41 格），執行它，會出現三個按鈕：

- **匯出 A 問答** → 把 A 的答案存成 txt。
- **匯出 B 問答** → 把 B 的答案存成 txt。
- **匯出對比 (MD)** → 把 A、B 並排對比存成 Markdown。

按鈕隨時可按。如果某邊還沒問答（沒資料），按了只會提示「沒有 output」，不會出錯。

存檔位置：
- Colab：`MyDrive/FJU_dongpoProject/dongPo/results`
- 本地：`E:\FJU_dongpoProject\notebook_from_0608\dongPo\results`

檔名範例：`A_20260608_191108.txt`、`dongpo_compare_20260608_191109.md`
（如果你開了 greedy 模式，會變成 `greedy_A_...txt`、`greedy_dongpo_compare_...md`）

---

## 常見狀況快速排解

| 你看到的狀況 | 可能原因 | 怎麼辦 |
|---|---|---|
| 載入卡在某個 % 不動、超久 | VRAM 不夠，被擠到一般記憶體 | 你的硬體不夠力，換更好的環境。這不是 bug。 |
| 報錯 `model type qwen3 ... not recognize` | transformers 版本太舊 | 本地請先跑「降版那格」再重啟 kernel |
| 報錯 `has no attribute float8...` | transformers 版本太新（5.x） | 同上，鎖回 4.51.3 |
| 換 B 時載入超慢 | A 沒清乾淨，顯存爆了 | 重啟 kernel，別硬撐 |
| 按存檔說「沒有 output」 | 那一邊還沒跑問答 | 先跑完該邊的問答再存 |
| 載入報「找不到 adapter / 底模」 | 模型沒放到指定資料夾 | 回去檢查路徑那段，把檔案放對位置 |
| 本地 vs Colab 答案不一樣 | 不同硬體、不同精度，正常現象 | 想穩定就開 greedy；跨環境本來就不保證逐字一致 |

---

## 一句話總結流程

**準備模型 → 照順序執行到「路徑驗證成功」→ 載入 A → 問 A → 清乾淨（重啟最穩）→ 載入 B → 問 B → 看比較 → 存檔。**

完。
