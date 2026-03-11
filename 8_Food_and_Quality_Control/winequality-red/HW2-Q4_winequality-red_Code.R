##HW2-Q4
#步驟一：讀取資料與資料預處理
# 1. 讀取資料
redwine <- read.csv("winequality-red.csv", header = TRUE)

# 2. 依照題目要求重新定義 rating
# poor: quality 3, 4, 5 | good: quality 6, 7, 8
redwine$rating <- ifelse(redwine$quality <= 5, "poor", "good")
redwine$rating <- factor(redwine$rating, levels = c("poor", "good")) #把poor/good正式轉化為R語言認得的類別資料(Factor)，"poor", "good"代表從 poor 變成 good 的機率是多少

# 3. 移除原本的 quality 欄位，避免干擾模型
wine_data <- redwine[, -12]

# 4. 切分訓練集與測試集
set.seed(123) # 固定隨機種子，讓結果可重複
#id <- sample(nrow(wine_data), 1000) #在wine_data的所有列中，隨機取1000筆資料
id=sample(1:nrow(wine_data), round(nrow(wine_data)/3)) #在wine_data的所有列中，取1/3的資料
tr.red <- wine_data[-id, ]   # 1/3以外的資料定義為訓練集
ts.red <- wine_data[id, ]  # 1/3的資料定義為測試集

#步驟二：建構邏輯迴歸模型 (Logit Model):找出「顯著的預測因子」（significant predictors）
# 建立邏輯迴歸模型: family = "binomial" 代表這是二元分類 (Logistic Regression)
logit_model <- glm(rating ~ ., data = tr.red, family = "binomial") #用Generalized Linear Model(廣義線性模型)，用tr.red資料裡除了目標(rating)以外的自變數來預測rating
# 查看模型結果，找出顯著因子
summary(logit_model)

#步驟三：使用顯著因子進行天真貝氏分類 (Naïve Bayes)
library(e1071)
# 使用 Logit 找出的顯著因子：揮發性酸度、總二氧化硫、硫酸鹽、酒精、檸檬酸
# 建立 Naive Bayes 模型
nb_model <- naiveBayes(rating ~ volatile.acidity + total.sulfur.dioxide + sulphates + alcohol + citric.acid + chlorides + free.sulfur.dioxide, data = tr.red)
print(nb_model)

#步驟四: 比較Logit Model和Naïve Bayes在 ts.red 上的表現
# 1. 邏輯迴歸預測: type="response" 會給出 0 到 1 之間的機率
logit_prob <- predict(logit_model, newdata = ts.red, type = "response")
logit_pred <- ifelse(logit_prob > 0.5, "good", "poor") # 如果機率 > 0.5 就是 good，否則就是 poor
logit_pred <- factor(logit_pred, levels = c("poor", "good"))

# 2. 單純貝氏預測
nb_pred <- predict(nb_model, newdata = ts.red)

# 3. 建立混淆矩陣
logit_tab <- table(Actual = ts.red$rating, Predicted = logit_pred)
nb_tab <- table(Actual = ts.red$rating, Predicted = nb_pred)

# 4. 印出結果
print("--- Logit Model ---")
print(logit_tab)
print(paste("Accuracy:", sum(diag(logit_tab))/sum(logit_tab)))

print("--- Naïve Bayes ---")
print(nb_tab)
print(paste("Accuracy:", sum(diag(nb_tab))/sum(nb_tab)))

## 視覺化--ROC 曲線
# install.packages("ROCR") 
library(ROCR)

#第一步：準備「預測機率」
# 1. 邏輯迴歸的機率:之前算過的logit_prob 已經是機率了
# 2. Naive Bayes 的機率 (要重新預測，要求它給出 raw 機率)
nb_prob_all <- predict(nb_model, newdata = ts.red, type = "raw")
# 我們只需要「預測是 good」的那一欄機率 (通常是第 1 或 第 2 欄，看你的類別排序)
nb_prob <- nb_prob_all[, "good"]

#第二步：計算圖表的座標點
# --- 邏輯迴歸的曲線資料 ---
pred_logit <- prediction(logit_prob, ts.red$rating, label.ordering = c("poor", "good"))
perf_logit <- performance(pred_logit, "tpr", "fpr") # tpr: 答對率, fpr: 誤判率

# --- Naive Bayes 的曲線資料 ---
pred_nb <- prediction(nb_prob, ts.red$rating, label.ordering = c("poor", "good"))
perf_nb <- performance(pred_nb, "tpr", "fpr")

#第三步：畫圖
# 先畫 Logit 的線 (紅色)
plot(perf_logit, col = "red", lwd = 2, main = "ROC Curve: Logit vs Naive Bayes")

# 再把 Naive Bayes 的線疊上去 (藍色)
plot(perf_nb, col = "blue", lwd = 2, add = TRUE)

# 畫一條 45 度的虛線 (代表瞎猜的基準線)
abline(a = 0, b = 1, lty = 2, col = "gray")

# 加上圖例
legend("bottomright", legend = c("Logit", "Naive Bayes"), 
       col = c("red", "blue"), lwd = 2)

# 在畫 perf_logit 時，加入 colorize = TRUE，標出門檻值機率0.5
plot(perf_logit, colorize = TRUE, print.cutoffs.at = 0.5, text.adj = c(-0.2, 1.7))

#####兩條線合併+0.5機率點位的圖######
# 1. 先畫 Logit 的線，並標註 0.5 的點
# print.cutoffs.at: 指定要標出的門檻值
# text.adj: 調整文字位置 (避免擋住線)
plot(perf_logit, col = "red", lwd = 2, main = "ROC Curve with 0.5 Threshold",
     print.cutoffs.at = 0.5, text.adj = c(-0.2, 1.7))

# 2. 疊加 Naive Bayes 的線，同樣標註 0.5
plot(perf_nb, col = "blue", lwd = 2, add = TRUE,
     print.cutoffs.at = 0.5, text.adj = c(1.2, -0.5))

# 3. 畫 45 度基準線
abline(a = 0, b = 1, lty = 2, col = "gray")

# 4. 加上圖例
legend("bottomright", legend = c("Logit (0.5 point)", "Naive Bayes (0.5 point)"), 
       col = c("red", "blue"), lwd = 2)

##計算AUC (Area Under the Curve)
# 計算 Logit 的 AUC
auc_logit <- performance(pred_logit, "auc")@y.values[[1]]

# 計算 Naive Bayes 的 AUC
auc_nb <- performance(pred_nb, "auc")@y.values[[1]]

# 印出來比比看
cat("Logit 的 AUC 分數: ", auc_logit, "\n")
cat("Naive Bayes 的 AUC 分數: ", auc_nb, "\n")
