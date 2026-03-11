---
title: Data science HW2-Q4Q8 Presentation

---

# HW2-Q4 Classifying Redwine Dataset using Logit model and Naive Bayes 
**紅酒品質分類預測報告：Logit 與 Naïve Bayes 模型比較**

## 項目摘要

本分析旨在使用紅酒品質資料集（Wine Quality Dataset），透過**邏輯迴歸（Logistic Regression）** 篩選關鍵影響因子，並對比 **天真貝氏分類器（Naïve Bayes）** 的預測效能。

---

## 步驟一：資料讀取與預處理

在建模之前，必須先清理資料。最重要的步驟是將品質分數（3-8分）轉化為二元分類「好酒（good）」與「普通酒（poor）」。

### 1. 讀取與標籤重新定義

題目將品質 3,4,5 分定義為 `poor`， 6,7,8 分定義為 `good`。並透過 `factor` 將其轉換為 R 語言識別的類別資料。

```r
redwine <- read.csv("winequality-red.csv", header = TRUE)

# 重新定義 rating: poor (3,4,5) | good (6,7,8)
redwine$rating <- ifelse(redwine$quality <= 5, "poor", "good")

# 轉化為 Factor，並設定排序基準-poor 是基礎（0），good 是目標（1）
redwine$rating <- factor(redwine$rating, levels = c("poor", "good")) 

# 移除原始 quality 欄位，避免模型產生遞迴關聯
wine_data <- redwine[, -12]
```

### 2. 切分訓練集與測試集

為了驗證模型準確性，我們設定隨機種子 `set.seed(123)` 確保結果可重複，並拆分1/3的資料作為測試集，其餘則作為訓練集。

```r
set.seed(123) 
id=sample(1:nrow(wine_data), round(nrow(wine_data)/3))  
tr.red <- wine_data[-id, ]   # 1/3以外的資料定義為訓練集
ts.red <- wine_data[id, ]  # 1/3的資料定義為測試集
```

---

## 步驟二：建構邏輯迴歸模型 (Logit Model)

我們使用廣義線性模型（GLM）來找出哪些化學成分對紅酒評分有「顯著影響」。這一步是為了找出**顯著預測因子（Significant Predictors）**。

```r
# 建立二元分類邏輯迴歸模型，用tr.red資料裡除了rating以外的自變數來預測rating
logit_model <- glm(rating ~ ., data = tr.red, family = "binomial") 

# 觀察 Summary 產出的 P-value (星號部分)
summary(logit_model)
```
```
Coefficients:
                       Estimate Std. Error z value Pr(>|z|)    
(Intercept)           20.413402  97.517746   0.209   0.8342    
fixed.acidity          0.130540   0.120509   1.083   0.2787    
volatile.acidity      -3.577976   0.622576  -5.747 9.08e-09 ***
citric.acid           -1.564403   0.703076  -2.225   0.0261 *  
residual.sugar         0.033620   0.064415   0.522   0.6017    
chlorides             -4.818592   2.031531  -2.372   0.0177 *  
free.sulfur.dioxide    0.024643   0.010194   2.417   0.0156 *  
total.sulfur.dioxide  -0.016649   0.003472  -4.795 1.63e-06 ***
density              -27.485853  99.538528  -0.276   0.7824    
pH                    -0.756515   0.884388  -0.855   0.3923    
sulphates              3.174661   0.568898   5.580 2.40e-08 ***
alcohol                0.922037   0.129424   7.124 1.05e-12 ***
```
---

## 步驟三：使用顯著因子進行天真貝氏分類 (Naïve Bayes)

根據步驟二的結果，我們挑選出具有統計顯著性（具有星號）的因子：**揮發性酸度、總二氧化硫、硫酸鹽、酒精、檸檬酸、氯化物、遊離二氧化硫**，並以此建立 Naïve Bayes 模型。

```r
library(e1071)

# 僅使用顯著因子建立模型
nb_model <- naiveBayes(rating ~ volatile.acidity + total.sulfur.dioxide + 
                        sulphates + alcohol + citric.acid + chlorides + 
                       free.sulfur.dioxide, data = tr.red)
print(nb_model)
```
```
Naive Bayes Classifier for Discrete Predictors

Call:
naiveBayes.default(x = X, y = Y, laplace = laplace)

A-priori probabilities:
Y
     poor      good 
0.4596623 0.5403377 

Conditional probabilities:
      volatile.acidity
Y           [,1]      [,2]
  poor 0.5896327 0.1711538
  good 0.4724566 0.1573669

      total.sulfur.dioxide
Y          [,1]     [,2]
  poor 54.98673 36.37493
  good 40.23785 29.20758

      sulphates
Y           [,1]      [,2]
  poor 0.6165714 0.1776617
  good 0.6983333 0.1644406

      alcohol
Y           [,1]      [,2]
  poor  9.918265 0.7632055
  good 10.862963 1.1052181

      citric.acid
Y           [,1]      [,2]
  poor 0.2389796 0.1853714
  good 0.3004514 0.2004859

      chlorides
Y            [,1]       [,2]
  poor 0.09308776 0.05485108
  good 0.08290799 0.03609269

      free.sulfur.dioxide
Y          [,1]     [,2]
  poor 16.55306 10.83140
  good 15.48785 10.12821

```
---

## 步驟四：模型表現比較 (Accuracy)

我們將兩個模型套用到未參與訓練的測試集（`ts.red`）上，觀察其混淆矩陣與整體準確度。

```r
# 1. 邏輯迴歸預測：type="response" 會給出 0 到 1 之間的機率
#    設定門檻值 > 0.5 就是 good，否則就是 poor
logit_prob <- predict(logit_model, newdata = ts.red, type = "response")
logit_pred <- factor(ifelse(logit_prob > 0.5, "good", "poor"), levels = c("poor", "good"))

# 2. 單純貝氏預測
nb_pred <- predict(nb_model, newdata = ts.red)

# 3. 建立混淆矩陣並計算 Accuracy
logit_tab <- table(Actual = ts.red$rating, Predicted = logit_pred)
nb_tab <- table(Actual = ts.red$rating, Predicted = nb_pred)

print("--- Logit Model Accuracy ---")
print(sum(diag(logit_tab))/sum(logit_tab))

print("--- Naïve Bayes Accuracy ---")
print(sum(diag(nb_tab))/sum(nb_tab))
```
```
[1] "--- Logit Model ---"
      Predicted
Actual poor good
  poor  180   74
  good   69  210
"Accuracy: 0.731707317073171"

[1] "--- Naïve Bayes ---"
      Predicted
Actual poor good
  poor  173   81
  good   61  218
"Accuracy: 0.733583489681051"
```
不分train / test 的結果：
```
[1] "--- Logit Model ---"
      pred
actual poor good
  good  646  209
  poor  193  551
"Accuracy:  0.7485929"

[1] "--- Naïve Bayes ---"
      pred
actual  good poor
  good  675  180
  poor  243  501
"Accuracy: 0.7354597"
```
不分群的表現也沒有拆分好多少

---

## 步驟五：視覺化評估 (ROC 曲線與 AUC)

為了更全面評估模型，我們繪製 ROC 曲線。這能幫助我們觀察模型在「答對率（TPR）」與「誤判率（FPR）」之間的權衡。

```r
library(ROCR)

# 準備預測機率
nb_prob <- predict(nb_model, newdata = ts.red, type = "raw")[, "good"]

# 計算曲線座標點：特別加入 label.ordering 明確指定poor和good的順序，確保曲線向上翻轉
pred_logit <- prediction(logit_prob, ts.red$rating, label.ordering = c("poor", "good"))
perf_logit <- performance(pred_logit, "tpr", "fpr")

pred_nb <- prediction(nb_prob, ts.red$rating, label.ordering = c("poor", "good"))
perf_nb <- performance(pred_nb, "tpr", "fpr")

# 繪製圖表
# 1. 先畫 Logit 的線 (紅色)，並標註 0.5 的點
# print.cutoffs.at: 指定要標出的門檻值
# text.adj: 調整文字位置以利閱讀
plot(perf_logit, col = "red", lwd = 2, main = "ROC Curve with 0.5 Threshold",
     print.cutoffs.at = 0.5, text.adj = c(-0.2, 1.7))

# 2. 疊加 Naive Bayes 的線 (藍色)，同樣標註 0.5
plot(perf_nb, col = "blue", lwd = 2, add = TRUE,
     print.cutoffs.at = 0.5, text.adj = c(1.2, -0.5))

# 3. 畫 45 度基準線 (代表隨機猜測的表現)
abline(a = 0, b = 1, lty = 2, col = "gray")

# 4. 加上圖例說明
legend("bottomright", 
       legend = c("Logit (0.5 point)", "Naive Bayes (0.5 point)"), 
       col = c("red", "blue"), lwd = 2)
```
![plot_zoom_png](https://hackmd.io/_uploads/BJ9QzU_P-g.png)

```
# 計算 AUC 數值 (Area Under the Curve)
auc_logit <- performance(pred_logit, "auc")@y.values[[1]]
auc_nb <- performance(pred_nb, "auc")@y.values[[1]]

cat("Logit 的 AUC 分數: ", auc_logit, "\n")
cat("Naive Bayes 的 AUC 分數: ", auc_nb, "\n")
```
```
Logit 的 AUC 分數:  0.8042503 
Naive Bayes 的 AUC 分數:  0.7976745 
```
---

## 結論與發現

![image](https://hackmd.io/_uploads/BknG9LdP-e.png)

1. **模型準確度分析 (Accuracy)**
   在預設門檻值為 0.5 的情況下，兩個模型的準確度旗鼓相當。雖然 Naïve Bayes 略高一些，但這僅代表在「0.5」這個單一切點上的表現。     
1. **ROC 曲線與 AUC 分數分析 (Model Stability)**
   雖然 Accuracy 相似，但 Logit 的 AUC (0.8043) 跨越了 0.8 的優良門檻。
*    ROC 曲線意涵：ROC 曲線是由無數個切點組成的。Logit 模型擁有較高的 AUC，說明其整體預測機率的排序更為精準。  
*    穩定性：Logit 模型在面對不同判斷標準（門檻）時，比 Naïve Bayes 展現了更穩健的分類界線。這是判斷模型「區分能力」最重要的指標。雖然 Naïve Bayes 的 Accuracy 略高，但 AUC 分數卻是 Logit 稍微領先。
3.  **模型選擇**：若目標是獲取較佳的總體準確率，建議優先選用邏輯迴歸模型。

---

# HW2-Q8 Container Dataset Forecasting Linear Regression & Biased Regression（Ridge / Lasso / Elastic Net）

## 一、研究目的（Objective）
- 使用 **Container dataset** 多個經濟與航運相關指標（X）來預測 **Container 營收指標（Y）**，並比較四種迴歸模型在 **測試集 2023–2025年** 的預測表現：
  - Multiple Linear Regression (MLR)
  - Ridge Regression （縮係數，不會變 0）
  - Lasso Regression （縮係數 + 部分係數變 0 → 自動篩選變數）
  - Elastic Net （折衷：保留更多變數但仍有篩選效果）

* **評估指標使用**：
  - RMSE（Root Mean Squared Error，均方根誤差）
  - MAE（Mean Absolute Error，平均絕對誤差）
  - MAPE（Mean Absolute Percentage Error，平均絕對百分比誤差）

---

## 二、資料前處理規則（依題目限制）
本題的資料處理完全按照題目限制條件：

1. 使用 `container.csv`
2. **刪除 HSI 欄位**（因缺失值太多）
3. **只保留前 178 筆** (刪除缺失值列****)
4. 訓練集：2011–2022年；測試集：2023–2025年
5. `Date` 轉成日期、並抽出 `Year` 做切分

---

## 三、 R 程式碼

```r
library(glmnet)
set.seed(1)

# 1) Load Data
container <- read.csv("container.csv", header = TRUE)
stopifnot("Date" %in% names(container))
stopifnot("Container" %in% names(container))

# 2) Drop HSI + take first 178 rows
if ("HSI" %in% names(container)) container$HSI <- NULL
container <- container[1:178, , drop = FALSE]

# 3) Parse Date -> Year, remove missing values
container$Date <- as.Date(container$Date, format = "%Y/%m/%d")
if (any(is.na(container$Date))) stop("Date parsing failed.")
container$Year <- as.numeric(format(container$Date, "%Y"))
container <- na.omit(container)

# 4) Train/Test split by year
train_df <- subset(container, Year >= 2011 & Year <= 2022)
test_df  <- subset(container, Year >= 2023 & Year <= 2025)
if (nrow(train_df) == 0 || nrow(test_df) == 0) stop("Train/Test split is empty.")

# 5) Prepare X/y (glmnet style)
drop_cols <- c("Container", "Date", "Year")
x_cols <- setdiff(names(container), drop_cols)

X_train <- data.matrix(train_df[, x_cols, drop = FALSE])
y_train <- train_df$Container
X_test  <- data.matrix(test_df[, x_cols, drop = FALSE])
y_test  <- test_df$Container

# 6) Metrics
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
mae  <- function(y, yhat) mean(abs(y - yhat))
mape <- function(y, yhat) mean(abs((y - yhat) / y))

score_table <- array(0, c(3, 4))
rownames(score_table) <- c("RMSE", "MAE", "MAPE")
colnames(score_table) <- c("MLR", "Ridge", "Lasso", "ElasticNet")

# 7) MLR
mlr_fit <- lm(Container ~ ., data = train_df[, c("Container", x_cols), drop = FALSE])
pred_mlr_test <- predict(mlr_fit, newdata = test_df[, x_cols, drop = FALSE])
score_table["RMSE", "MLR"] <- rmse(y_test, pred_mlr_test)
score_table["MAE",  "MLR"] <- mae(y_test, pred_mlr_test)
score_table["MAPE", "MLR"] <- mape(y_test, pred_mlr_test)

# 8) glmnet via CV (lambda.min)
K <- 5
foldid <- sample(rep(1:K, length.out = nrow(X_train)))

cv_ridge <- cv.glmnet(X_train, y_train, family="gaussian", alpha=0,   foldid=foldid)
cv_lasso <- cv.glmnet(X_train, y_train, family="gaussian", alpha=1,   foldid=foldid)
cv_enet  <- cv.glmnet(X_train, y_train, family="gaussian", alpha=0.5, foldid=foldid)

pred_ridge_test <- as.numeric(predict(cv_ridge, newx=X_test, s="lambda.min"))
pred_lasso_test <- as.numeric(predict(cv_lasso, newx=X_test, s="lambda.min"))
pred_enet_test  <- as.numeric(predict(cv_enet,  newx=X_test, s="lambda.min"))

score_table["RMSE", "Ridge"] <- rmse(y_test, pred_ridge_test)
score_table["MAE",  "Ridge"] <- mae(y_test, pred_ridge_test)
score_table["MAPE", "Ridge"] <- mape(y_test, pred_ridge_test)

score_table["RMSE", "Lasso"] <- rmse(y_test, pred_lasso_test)
score_table["MAE",  "Lasso"] <- mae(y_test, pred_lasso_test)
score_table["MAPE", "Lasso"] <- mape(y_test, pred_lasso_test)

score_table["RMSE", "ElasticNet"] <- rmse(y_test, pred_enet_test)
score_table["MAE",  "ElasticNet"] <- mae(y_test, pred_enet_test)
score_table["MAPE", "ElasticNet"] <- mape(y_test, pred_enet_test)

score_table

```

## 四、Multiple Linear Regression（MLR）變數分析

本節針對 **Multiple Linear Regression（MLR，多元線性迴歸）** 模型之估計係數進行分析，說明各解釋變數對 Container 營收指標的影響方向、統計顯著性，以及其潛在的管理意涵。

---

### 4.1 模型整體解釋力

由 MLR 結果可得：

- Multiple R-squared = **0.9923**
- Adjusted R-squared = **0.9913**
- F-statistic p-value < **2.2e-16**

顯示模型在訓練期間（2011–2022）具有**極高的解釋能力**，整體而言，所選取的經濟、貿易與航運相關指標能有效解釋 Container 營收指標的變動。

然而，由於模型中包含多個高度相關的宏觀與營收指標，單一係數之解讀仍需留意 **Multicollinearity（共線性）** 的可能影響。

---
### 4.2 核心變數說明與解讀（p-value < 0.05）

以下針對 MLR 模型中影響 Container 營收指標最顯著的四個變數進行說明與解釋：

---

#### (1) GCTI（全球貨櫃吞吐量指數 Global Container Throughput Index）

**指標意義**  
GCTI 為全球貨櫃吞吐量指數，用來反映全球主要港口的貨櫃處理量與航運活動熱度，可視為整體航運需求與物流繁忙程度的代表。

**模型結果（正向，＊＊＊）**    
- 係數方向：正向  
- 統計顯著（p < 0.001）

**解讀**  
當全球貨櫃吞吐量增加，代表國際貿易與物流活動活絡，市場對運輸服務的需求上升，在運力有限的情況下，Container 營收指標自然容易上揚。

**小結**  
>「GCTI 顯著為正，顯示全球航運需求越熱絡，Container 運價越高，反映需求端對運價的推升效果。」

---

#### (2) CCFI（中國出口集裝箱運價指數 China Containerized Freight Index）

**指標意義**  
CCFI 為中國出口貨櫃運價指數，直接反映中國對外航線的貨櫃運價水準，是市場上最具代表性的實際運價指標之一。

**模型結果（正向，＊＊＊）**  
- 係數方向：正向  
- 統計高度顯著（p < 0.001）

**解讀**  
CCFI 本身即為貨櫃運價指標，其上升通常代表出口需求強勁或航線供需吃緊，因此與 Container 營收指標呈現高度正向關聯。

**小結**  
>「CCFI 與 Container 運價呈高度正向且顯著的關係，顯示航運市場本身的運價水準是影響 Container 營收指標的最直接因素之一。」

---

#### (3) Container_SeasonIndex（貨櫃運價季節性指數）

**指標意義**  
Container_SeasonIndex 為季節性指標，用來描述航運市場在不同時間點（例如旺季、淡季）的結構性差異，如年終出貨旺季、補庫存週期等。

**模型結果（正向，＊＊＊）**  
- 係數方向：正向  
- 統計顯著（p < 0.001）  


**解讀**  
航運市場具有明顯季節性特徵，旺季期間需求集中、艙位緊張，容易推升運價；淡季則相反，因此季節性對 Container 營收指標的解釋力極高。

**小結**  
>「季節性指標顯著且係數大，顯示 Container 營收指標高度依賴旺季與淡季結構，是影響運價波動的重要基礎因素。」

---

#### (4) WTI.crude.Oil（西德州原油價格）

**指標意義**  
WTI 原油價格為全球重要能源價格指標，常被視為運輸成本與整體景氣循環的代表變數。

**模型結果（負向，＊＊＊）**  
- 係數方向：負向  
- 統計顯著（p < 0.001）

**解讀**  
在控制其他航運與貿易指標後，油價上升與 Container 營收指標呈現顯著負向關係，可能反映油價上升對需求端造成壓力，或在景氣循環中扮演反向訊號角色。

本研究僅將此結果解讀為統計上的關聯，而非直接的因果推論。

****  
>「WTI 原油價格在模型中呈現顯著負向，可能反映油價上升時需求或景氣面臨壓力，因此對 Container 營收指標形成抑制效果，但此處僅作關聯解釋。」



---

### 4.3 邊緣顯著變數（0.05 ≤ p-value < 0.1）

#### Retail.sales.index_US（正向，．）
- 係數為正，但僅達邊緣顯著（p = 0.096）
- 可能存在一定關聯，但影響穩定性不足
- 需進一步資料或模型設計（如 lag variables）驗證其效果

---

### 4.4 未達顯著變數（p-value ≥ 0.1）

包含 SCFI、IPI_TW、Retail.sales.index_EU、Import_chemical、Export_chemical、Import_.base.metals、Import_vegetable.of.products、Export_.plastics 等變數，在本模型中未達統計顯著水準。

可能原因包括：
1. 與其他運價指標（如 CCFI、GCTI）高度相關，產生共線性問題  
2. 與 Container 之關係可能具非線性或時間落後效果  
3. 在本研究期間內，其影響不具一致性

### 表1：MLR 係數顯著性彙整與解釋（Training: 2011–2022）

| 變數 | 方向 | p-value | 一句話解釋 |
|---|---|---:|---|
| GCTI | 正 | 2.33e-12 *** | 全球航運景氣指標上升時，Container 營收指標顯著上升，反映航運景氣對運價的正向影響。 |
| CCFI | 正 | < 2e-16 *** | 中國航運運價指數與 Container 呈高度正向關聯，是最穩定的營收指標驅動因子之一。 |
| SCFI | 負 | 0.103 | 與 Container 關係不顯著，可能與其他指標高度相關而產生共線性。 |
| IPI_TW | 負 | 0.332 | 台灣工業生產指數在控制其他變數後，對 Container 營收指標影響不明顯。 |
| Retail.sales.index_EU | 負 | 0.517 | 歐洲零售銷售指數與 Container 之關聯在本模型中不具統計顯著性。 |
| Retail.value_CN | 正 | 0.018 * | 中國零售活動增加可能帶動貨物流通，進而推升 Container 營收指標。 |
| Export_.basemetals | 正 | 0.002 ** | 工業原物料出口增加代表貿易與運輸需求上升，與 Container 呈顯著正向關係。 |
| Import_chemical | 負 | 0.168 | 化學品進口對 Container u/營收指標影響不顯著，可能被其他貿易指標吸收。 |
| Retail.sales.index_US | 正 | 0.096 . | 美國零售銷售與運價呈正向但僅達邊緣顯著，影響穩定性有限。 |
| Container_SeasonIndex | 正 | 0.000393 *** | Container 營收指標高度受到季節性影響，旺季效應為主要波動來源之一。 |
| Export_chemical | 負 | 0.659 | 化學品出口在本模型中未顯示穩定的運價解釋力。 |
| Import_.base.metals | 負 | 0.561 | 基礎金屬進口與 Container 營收指標之關係不顯著。 |
| Import_vegetable.of.products | 正 | 0.142 | 農產品進口與運價呈正向但未達顯著水準。 |
| WTI.crude.Oil | 負 | 0.000246 *** | 油價上升在控制其他變數後與 Container 營收指標呈顯著負向關聯，可能反映需求或景氣壓力。 |
| Export_.plastics | 正 | 0.328 | 塑膠製品出口對 Container 營收指標影響不明顯。 |
| Import_machine | 正 | 0.0059 ** | 機械設備進口增加代表投資與貿易活動擴張，進而推升運輸需求與 Container 營收指標。 |

註：  
*** p < 0.001，** p < 0.01，* p < 0.05，. p < 0.1



### 4.5 MLR KPI總結

綜合 MLR 係數結果，可歸納以下重點：

1. **航運景氣指標（CCFI、GCTI）為 Container 營收指標最穩定且顯著的驅動因子**
2. **季節性因素對 Container 營收指標影響大，旺季與淡季效應明顯**
3. **需求與貿易活動（機械進口、中國零售、工業原料出口）與 Container 營收指標呈正向關聯**
4. **油價（WTI）可視為潛在風險觀察指標，其上升可能對 Container 營收指標形成壓力**

整體而言，Container 營收指標主要受到航運景氣、季節性與貿易需求共同影響。





## 五、CV 選出來的 lambda 與保留變數數量（Nonzero）

本次 Ridge / Lasso / ElasticNet 使用 **Cross-validation（交叉驗證）** 的 `cv.glmnet()`，以 **lambda.min** 作為最終模型的正則化強度（regularization strength）。

> 註：同一組 `foldid` 用於三個模型，確保比較公平。

### 5.1 CV 結果整理
lambda.min 是「CV 誤差最小的 lambda」
lambda.1se 是「在 1 個標準差內、最保守（最大）的 lambda」

| Model | alpha | lambda.min | lambda.1se | Nonzero（lambda.min） | 解讀 |
|---|---:|---:|---:|---:|---|
| Ridge | 0 | 94,762,196 | 150,888,214 | 16 | Ridge 只會「縮小」係數，不會變 0，因此仍保留全部變數 |
| Lasso | 1 | 8,241,933 | 13,123,487 | 11 | Lasso 會把部分係數壓到 0，達到 **Feature selection（特徵篩選）** |
| ElasticNet| 0.5 | 2,564,356 | 16,483,865 | 14 | 介於 Ridge 與 Lasso 之間：部分縮係數 + 部分篩選 |

**重點觀察：**
- Ridge 的 `lambda.min` 非常大（~9.48e7），代表正則化效果強、迴歸係數被大幅收縮，模型較為保守，較容易出現 **underfitting**。
- Lasso 最稀疏（Nonzero=11），模型更精簡，但可能犧牲部分預測精準度。
- ElasticNet 介於兩者之間（Nonzero=14），通常是「穩定 + 不過度稀疏」的折衷。

>結論來看：當lambda 偏小 → 懲罰弱 → 正則化模型會趨近 MLR。

---

## 六、測試集（2023–2025）績效比較

以下指標皆以 Test（2023–2025）計算，觀察泛化能力：

| Metric | MLR | Ridge | Lasso | ElasticNet |
|---|---:|---:|---:|---:|
| RMSE (均方根誤差) | 1.986014e+08 | 3.938162e+08 | 2.033081e+08 | 1.978514e+08 |
| MAE (M平均絕對誤差) | 1.274282e+08 | 3.474188e+08 | 1.559257e+08 | 1.439138e+08 |
| MAPE (平均絕對百分比誤差) | 6.8621％ | 20.9258% | 8.8815%| 8.1764% |

RMSE/MAE/MAPE 都是「預測誤差」，越小代表預測越準；MAPE 是百分比誤差；RMSE 對大錯誤更敏感；MAE 是最直觀的平均錯多少。

MAPE 錯誤極高：
- MLR = 6.86%
- Ridge = 20.93% （明顯誤差）
- Lasso = 7.92%
- ElasticNet = 8.06%

### 6.1 視覺化圖表


![HW2_Q8_plot(2011-2025)](https://hackmd.io/_uploads/ryLBIbdD-l.png)

#### 圖表解讀：Actual vs MLR / Ridge / Lasso / ElasticNet

> 圖中顏色對應（依 legend）：  
> **Actual（黑）**、**MLR（紅）**、**Ridge（藍）**、**Lasso（綠）**、**ElasticNet（紫）**  
> 分割線右側灰底為 **Test（2023–2025）**；左側為 **Train（2011–2022）**

**主要觀察：**
1. **Train 期間（2011–2022）四個模型整體都貼近 Actual**
   - 代表這組 KPI（如季節性、需求、油價、零售等）確實能解釋 Container 營收指標的大方向。
2. **2020–2022 附近出現大幅度飆升與急跌（劇烈波動 / 結構轉折）**
   - 這類「市場結構變動（structural change）」最容易造成模型誤差放大：  
     因為模型是用歷史規律學習，遇到制度/供需結構改變時容易跟不上。
3. **進入 Test（2023–2025）後，模型差異開始明顯**
   - Ridge（藍）與 Actual 的距離明顯變大（較常高估），顯示其在測試期的偏差較大。
   - MLR / Lasso / ElasticNet 大致仍能跟住趨勢，但在尖峰與急跌處仍會有落差。


![HW2_Q8_plot_TEST_2023_2025](https://hackmd.io/_uploads/BJBRBbODbx.png)


---

## 七、結果分析

### 7.1 哪個模型最好？（依不同評估指標）

- **RMSE 最小：ElasticNet（1.978514e+08）**  
  - 相對 MLR（1.986014e+08）只小約 **0.38%**（差距非常小）  
  - 代表 ElasticNet 對「少數大誤差」可能略有幫助，但提升有限

- **MAE 與 MAPE 最小：MLR（MAE=1.274282e+08、MAPE=6.8621％）**  
  - 代表 MLR 在「一般日常誤差」與「相對誤差」上最穩、最好解釋

> 結論：  
> - 若你的目標是「平均誤差最小、整體最穩」→ **MLR 最佳**  
> - 若你的目標是「降低少數極端誤差（RMSE）」→ **ElasticNet 略勝，但優勢很小**

---

### 7.2 為什麼 Ridge 在 Test 表現特別差？

Ridge 的 CV 選到 **非常大的 lambda.min（~9.48e7）**，代表正則化過強，係數被大量壓縮，導致模型變得過於保守，產生 **underfitting**。

直接反映在 Test 指標：
- RMSE：3.938e+08（遠高於 MLR 的 1.986e+08）
- MAE：3.474e+08（遠高於 MLR 的 1.274e+08）
- MAPE：0.209（遠高於 MLR 的 0.069）

> 白話：Ridge 這次「縮太多」，模型抓不到運價的波動幅度，所以預測偏差變大。

---

### 7.3 Lasso / ElasticNet 的角色與意義

- **Lasso（Nonzero=11）**：  
  具備特徵篩選能力，模型更精簡，但 Test 的 MAE/MAPE 明顯大於 MLR（平均誤差較大）。

- **ElasticNet（Nonzero=14）**：  
  在「保留較多變數 + 避免過度稀疏」間取得折衷，因此 RMSE 最小；但 MAE/MAPE 仍不如 MLR。

---

## 八、總結與建議

### 8.1 若以「預測穩定&誤差最小」為主要目標選擇「MLR」做預測
- **MLR** 在 MAE / MAPE 績效最佳，代表平均預測最穩、相對誤差最小，且係數解釋性最直觀。

### 8.2 本資料集不適合使用「偏差迴歸」做預測
- Ridge 在本次 CV 下選到過強的 lambda，導致underfitting、測試表現最差。  
- 本資料集不適合使用「偏差迴歸」做預測，MLR 績效明顯比三種偏差迴歸模型都好。
### 核心原因有三個（結構性）
* 這是「金額型（level）」時間序列，不是比例或高噪音橫斷面資料
   * 目標是：Sales revenue（營收，金額）
   * 營收的特性是：
       1. 數值量級大
       2. 明顯時間趨勢（trend）
       3. 波動主要來自整體環境影響
    
>    這類資料的主要變異，通常受幾個特定強趨勢因子影響，弱化了小變數的解釋力。
    
###  從結果來看：
1. Lasso 的「變數篩選能力」幾乎沒有發揮空間
2. ElasticNet 的「折衷優勢」也不明顯
3. Ridge 的 shrinkage 反而可能在 regime change 時放大偏差（明顯的波段test結果高估
---
## 九、補充說明
* ### Ridge/Lasso/ElasticNet Lambda=1
![messageImage_1770695481719](https://hackmd.io/_uploads/S1UoHVuv-g.jpg)

* ### Ridge/Lasso/ElasticNet Lambda=100000
![messageImage_1770695428363](https://hackmd.io/_uploads/SJaxIV_w-g.jpg)




> ### 小結：此資料集使用偏差迴歸做預測時，當 Lambda 設置越高（懲罰項越多，迴歸係數壓縮），test 誤差會越明顯。


-----







