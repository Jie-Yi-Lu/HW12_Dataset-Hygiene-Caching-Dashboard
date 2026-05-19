"""Generate a deliberately messy Stroop-like dataset for the Week 12 HOMEWORK.

執行：
    python generate_messy_stroop_homework.py

產出：messy_stroop_homework.csv (n=240 trials)

這份檔案的 **abnormalities 與課堂 demo (messy_stroop.csv) 不同** —
原則相同（dtype 錯誤、sentinel value、categorical 不一致、numeric 超出範圍），
但具體的「髒」樣態學生必須自己用 descriptive statistics 找出來，
**不能直接複製課堂 demo 的 clean() 函式**。

故意製造的問題（讓學生用本週原則發現並修補）：

    1. rt_ms 欄是字串 dtype（含 "missing" 與 "--" 兩種字串 sentinel）
    2. rt_ms 含 -1（負值 sentinel）與 9999（正值 sentinel）— 兩個方向都有
    3. condition 4 個 level 但其實 2 個（含尾端空白 / 縮寫 / 大小寫不一致）
    4. age 同時有 -1 與 888 兩種 sentinel（同一欄多種 sentinel — 真實常見）
    5. accuracy 混入字串 "True" / "False"（部分受試者用了不同編碼）
    6. subject_id × trial_id 有重複 trial（資料合併失誤的常見後果）
    7. Stroop effect 仍存在但量級為 +80ms（不同於 demo 的 +60ms）

⚠️  本檔案是 **homework 用**，學生會看到這個生成器 — 但他們應該被作業
要求把生成器當作 A.0 的「來源 1」自己讀過、紀錄發現的問題。
"""

import numpy as np
import pandas as pd

np.random.seed(2026)

N = 240   # 與課堂 demo (200) 不同，避免學生 hardcode

# ---------------------------------------------------------------
# Base dataset
# ---------------------------------------------------------------
df = pd.DataFrame({
    "trial_id":   np.arange(1, N + 1),                          # 新欄位：trial 編號
    "subject_id": np.random.choice([101, 102, 103, 104, 105, 106], N),  # ID 編碼不同
    "condition":  np.random.choice(
        ["congruent", "congruent ", "incongruent", "Incongruent"], N),  # 含尾端空白
    "rt_ms":      np.random.normal(520, 90, N),                 # mean/SD 略不同
    "accuracy":   np.random.choice([0, 1, 1, 1, 1], N),
    "age":        np.random.choice([22, 26, 30, 34, 38, 42, 46, 50, 54,
                                    58, 62, 66, 70, 74, -1, 888, np.nan], N),
})

# ---------------------------------------------------------------
# Inject Stroop effect first (before any messiness), magnitude 80 ms
# ---------------------------------------------------------------
incong_mask = df["condition"].str.strip().str.lower().isin(
    ["incongruent"]
)
df.loc[incong_mask, "rt_ms"] = df.loc[incong_mask, "rt_ms"] + 80

# ---------------------------------------------------------------
# Inject string sentinels into rt_ms — TWO different ones
# ---------------------------------------------------------------
rt_col = df["rt_ms"].astype(object)

# "missing" — 8 rows
missing_idx = np.random.choice(N, 8, replace=False)
rt_col.iloc[missing_idx] = "missing"

# "--" — 5 rows (different sentinel; tests if student handles multiple)
dash_idx = np.random.choice([i for i in range(N) if i not in missing_idx],
                            5, replace=False)
rt_col.iloc[dash_idx] = "--"

# ---------------------------------------------------------------
# Inject numeric sentinels into rt_ms — both directions
# ---------------------------------------------------------------
# -1 (negative sentinel — sometimes used for "no response")
remaining = [i for i in range(N)
             if i not in missing_idx and i not in dash_idx]
neg_idx = np.random.choice(remaining, 4, replace=False)
rt_col.iloc[neg_idx] = -1

# 9999 (positive sentinel — different from demo's 99999, easier to miss)
remaining = [i for i in remaining if i not in neg_idx]
pos_idx = np.random.choice(remaining, 4, replace=False)
rt_col.iloc[pos_idx] = 9999

df["rt_ms"] = rt_col

# ---------------------------------------------------------------
# Add extra condition spelling variants — "con" abbreviation + period
# ---------------------------------------------------------------
abbrev_idx = np.random.choice(N, 6, replace=False)
df.loc[abbrev_idx[:3], "condition"] = "con"
df.loc[abbrev_idx[3:], "condition"] = "incong."

# ---------------------------------------------------------------
# Mix string True/False into accuracy (some subjects logged differently)
# ---------------------------------------------------------------
acc_col = df["accuracy"].astype(object)
tf_idx = np.random.choice(N, 10, replace=False)
# Half "True", half "False"
for i, k in enumerate(tf_idx):
    acc_col.iloc[k] = "True" if i % 2 == 0 else "False"
df["accuracy"] = acc_col

# ---------------------------------------------------------------
# Inject duplicate trials (subject merging error)
# Pick 3 random rows and duplicate them (so trial_id repeats but row content
# is otherwise identical → a true duplicate row pattern).
# ---------------------------------------------------------------
dup_indices = np.random.choice(N, 3, replace=False)
duplicates = df.iloc[dup_indices].copy()
df = pd.concat([df, duplicates], ignore_index=True)

# Shuffle so duplicates aren't trivially at the bottom
df = df.sample(frac=1.0, random_state=2026).reset_index(drop=True)

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
out = "messy_stroop_homework.csv"
df.to_csv(out, index=False)
print(f"Wrote {out}  (n={len(df)} rows including duplicates)")
print()
print(df.head(8))
print()
print(f"dtype of rt_ms: {df['rt_ms'].dtype}")
print(f"dtype of accuracy: {df['accuracy'].dtype}")
print(f"condition levels: {sorted(df['condition'].unique().tolist())}")
print(f"age unique: {sorted([x for x in df['age'].dropna().unique()])}")
print(f"n duplicate trial_id: {df['trial_id'].duplicated().sum()}")
