# 金融與經濟分析專案作業整理

本文件梳理了 GitHub 專案中「金融與經濟分析 (Finance & Economics)」資料夾的所有作業 (HW) 與練習題 (Unit)。

---

## 📊 資料集 1：Asia_index (亞洲股市指數預測與風險)

* **【HW1】時間序列基礎預測：** 使用簡單移動平均 (n=10, 20, 30) 預測日本指數；使用指數平滑 (λ=0.25, 0.35, 0.45) 預測印度指數。計算 RMSE, MAE, MAPE 並視覺化（需包含 2024 資料）。
* **【HW2 - A】迴歸模型與 KPI 篩選：** 以 MLR 與 MARS 預測韓國或日本，取兩者 KPI 交集作為特徵，並最佳化 KNN 模型。比較三模型績效（訓練集 2011-2022，測試集 2023-2025）。
* **【HW2 - B】統計檢定：** 比較台、印、日三國的半年度報酬與風險，先進行 F 檢定再做 T 檢定。
* **【HW3】進階機器學習與因果解釋：** 以 XGB 與 MARS 預測日本或印度，取兩者 KPI 聯集作為特徵。比較 XGB, GB, MARS 績效（訓練集 2011-2021，測試集 2022-2024）。

## 📊 資料集 2：Bank (銀行行銷與授信分析)

* **【HW1】關聯規則分析：** 年齡離散化，數值欄位以中位數二分法處理。找出目標 "y" 的顯著規則並依頻率排序特徵。
* **【HW2 - A】邏輯斯迴歸與貝氏分類：** 建構 Logit 模型找出 p<5% 變數，比較單純貝氏分類器在「精簡特徵」與「所有特徵」下的績效 (70/30 切分)。
* **【HW2 - B】統計檢定：** 檢定年齡層餘額差異 (T-test) 與貸款比例差異 (Proportion test)。
* **【HW3】集成學習：** 透過卡方與 T 檢定篩選 KPI，建構 RF, AB, GB 模型。進行新樣本的預測演練。
* **【Unit 6】敘述統計與假設檢定：** 運用 ANOVA 與卡方檢定分析客戶輪廓與貸款意願之關聯。
* **【Unit 9】進階顧客行為預測：** 運用 Bagging, AdaBoost, XGBoost 預測定期存款，透過 ROC 曲線評估模型準確度。

## 📊 資料集 3：ETF (投資組合優化)

* **【HW1】基礎投資組合：** 挑選 00850 成分股，比較最大報酬、最小風險、最大風險報酬差模型之表現。
* **【HW4】數學規劃策略比較：** 比較五種策略（最大報酬、最小風險、平衡型、定期定額、等權重）之績效（2年訓練，半年測試）。
* **【HW5】非線性規劃與強化學習：** 基於夏普值挑選 0051 成分股，應用非線性規劃與強化學習進行投資組合優化 (2011-2022 訓練，2023-2025 測試)。

## 📊 資料集 4：usconsumption (總體經濟消費)

* **【Unit 14】時間序列預測：** 執行單根檢定 (ADF, PP, KPSS) 檢測穩態，利用 auto.arima 進行模型選取與未來 4 期預測。

# Finance & Economics Analysis Project Assignments

This document outlines the assignments (HW) and practice units (Unit) for the "Finance & Economics" folder in your GitHub repository.

---

## 📊 Dataset 1: Asia_index (Stock Index Forecasting & Risk)

* **【HW1】Time Series Basics:** Apply simple moving average (n=10, 20, 30) for "Japan" and exponential smoothing (λ=0.25, 0.35, 0.45) for "India". Visualize results and evaluate performance (RMSE, MAE, MAPE). Include 2024 data.
* **【HW2 - A】Regression & KPI Selection:** Predict "Korea" or "Japan" using MLR and MARS. Use the intersection of identified KPIs as input features. Apply KNN regression (optimized K) and compare MLR, MARS, and KNN performance (2011-2022 train, 2023-2025 test).
* **【HW2 - B】Statistical Testing:** Compare semiannual return and risk between Taiwan, India, and Japan. Perform F-test prior to T-test.
* **【HW3】Advanced ML & Causality:** Predict "Japan" or "India" using XGB and MARS. Use the union of identified KPIs as features. Compare performance of XGB, GB, and MARS (2011-2021 train, 2022-2024 test).

## 📊 Dataset 2: Bank (Marketing & Credit Analysis)

* **【HW1】Association Rules:** Discretize age (young <35, adult 35-55, old >55) and median-split numeric columns. Identify significant rules for target "y" and prioritize features by frequency.
* **【HW2 - A】Logit & Naïve Bayes:** Construct Logit models (p < 5%) to identify predictors. Compare Naïve Bayes performance (subset features vs. all) on 70/30 split.
* **【HW2 - B】Statistical Testing:** T-test (alternative="less") for balance differences between age groups; proportion test for loan ownership between single/non-single groups.
* **【HW3】Ensemble Learning:** Select KPIs via Chi-square and T-test. Construct RF, AB, and GB models (70/30 split). Predict for a "median/mode" sample.
* **【Unit 6】Descriptive Stats & Inference:** Analyze the relationship between customer profiles and loan behavior using Chi-square and ANOVA.
* **【Unit 9】Advanced Behavior Prediction:** Use Bagging, AdaBoost, and XGBoost to predict deposits. Evaluate via ROC curves.

## 📊 Dataset 3: ETF (Portfolio Optimization)

* **【HW1】Basic Portfolio:** Select 15 stocks from 00850 (since 2016). Compare Max Return, Min Risk, and Max (Return-Risk) models.
* **【HW4】Mathematical Programming:** Compare five strategies (Max Return, Min Risk, Balanced, DCA, Equal Weighted) using 00850 data (2 years train, 6 months test).
* **【HW5】Nonlinear & Reinforcement Learning:** Select top 15 stocks from 0051 (by Sharpe Ratio). Compare portfolios using nonlinear programming and RL (2011-2022 train, 2023-2025 test).

## 📊 Dataset 4: usconsumption (Macroeconomics)

* **【Unit 14】Time Series Forecasting:** Test stationarity (ADF, PP, KPSS). Model using auto.arima based on ACF/PACF analysis. Forecast future 4 periods.
