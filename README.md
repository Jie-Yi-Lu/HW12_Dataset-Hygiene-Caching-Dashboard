# Week 12 Homework — Dataset Hygiene & Caching Dashboard


## 如何執行

```bash
pip install pandas numpy matplotlib streamlit
streamlit run app.py
```

資料檔須位於 `./data/messy_stroop_homework.csv`。  
若遺失，在 `./data/` 執行：

```bash
python generate_messy_stroop_homework.py
```


## 資料來源

- 資料集：`./data/messy_stroop_homework.csv`（`seed=2026`）
- 文獻引用：Lamers, M. J., Roelofs, A., & Rabeling-Keus, I. M. (2010). Selective attention and response set in the Stroop task. *Memory & Cognition*, *38*(7), 893–904.

## 三條最重要的 Cleaning 決定

1. **`rt_ms` 合理範圍設為 200–2000 ms**：依據 Lamers et al. (2010) 的 outlier exclusion 標準；低於 200 ms 視為 anticipation，高於 2000 ms 視為 lapse。
2. **`rt_ms` 含四種 sentinel，需分兩步處理**：字串 `"missing"` 與 `"--"` 用 `pd.to_numeric(errors="coerce")` 處理；數值 `-1` 與 `9999` 用 `replace()` 處理 — 依據 generator 第 62–82 行。
3. **`condition` 六種變體統一為兩種**：`strip()` + `lower()` + `replace({"con": "congruent", "incong.": "incongruent"})` — 依據 generator 第 40–41、89–91 行與資料觀察。


## 為何 `outlier_sd` 放在 `analyse()` 而非 `clean()`

`clean()` 只處理資料品質問題（dtype 錯誤、sentinel、違反 schema 定義的值），這些由資料來源的規格決定，與研究問題無關。`outlier_sd` 是分析層面的決策，不同研究問題可能對同一份乾淨資料套用不同閾值。若放進 `clean()`，會將某個分析選擇永久寫死進資料，後續無法比較不同閾值的結果，違反 cleaning 與 analysis 的邊界紀律。