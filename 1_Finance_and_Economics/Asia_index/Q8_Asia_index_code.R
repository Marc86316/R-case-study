library(tidyverse)

Asia_index <-read.csv("Asia_index.csv")
Asia_index$Date <- as.Date(Asia_index$Date, format="%Y/%m/%d")
# 切分訓練集 (2011-2021)
train_data <- Asia_index %>% filter(Date >= "2011-01-01" & Date <= "2021-12-31")
# 切分測試集 (2022-2025)
test_data <- Asia_index %>% filter(Date >= "2022-01-01") 

#####找預測Japan的KPI##########################
#####################eXtreme GB##########
library(xgboost)
# A.設定目標變數 (Y) 與特徵 (X)
target_var <- "Japan"
# 排除 Date 和 Japan 本身，其餘皆為候選特徵
feature_vars <- setdiff(names(train_data), c("Date", target_var))

# B.轉換為 XGBoost 專用的矩陣格式
dtrain <- xgb.DMatrix(data = as.matrix(train_data[, feature_vars]), 
                      label = train_data[[target_var]])

# C. 執行 XGBoost 模型以篩選 KPI
# 參數設定：使用回歸模型 (reg:squarederror)
params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 6
)
set.seed(42)
model_xgb <- xgb.train(params = params, data = dtrain, nrounds = 100)

# D. 輸出特徵重要性 (找出 KPI)
xgb_features <- xgb.importance(feature_names = feature_vars, model = model_xgb)
# 顯示前 10 名 KPI
print(head(xgb_features, 10))

#####################MARS##########
library(earth)
# A.建立 MARS 模型
# degree=2 允許變數間的交互作用 (Interaction)
set.seed(42)
mars_model <- earth(x = train_data[,feature_vars], 
                    y = train_data[[target_var]], 
                    degree = 2)

# B.查看 MARS 模型摘要與變數重要性
summary(mars_model)
mars_imp <- evimp(mars_model) # Estimate Variable IMPortance
print("MARS Variable Importance:")
print(mars_imp)

# C.篩選出 MARS 認為有影響力的變數 (nsubsets > 0 或 gcv > 0)
mars_features <- rownames(mars_imp)[mars_imp[,"nsubsets"] > 0]
print("MARS Selected Features:")
print(mars_features)

#####################取KPI聯集##########
union_features <- c("NASDAQ", "USD", "India", "DJI", "SP500", 
                    "SOX", "Taiwan", "Korea", "SHI","gold")
#####⭐建模預測Japan & 比較###################
# A.設定目標變數 (Y)
target_jp <- "Japan"

# B.切分訓練集 (2011-2021) 與 測試集 (2022-2024)
train_jp <- Asia_index %>% 
  filter(Date >= "2011-01-01" & Date <= "2021-12-31") %>%
  select(all_of(union_features), all_of(target_jp))

test_jp <- Asia_index %>% 
  filter(Date >= "2022-01-01") %>%
  select(all_of(union_features), all_of(target_jp))

# C.建立對照表
tableJP = array(0, c(3,3))
rownames(tableJP)<-c("RMSE","MAE","MAPE")
colnames(tableJP)<-c("XGB", "GB", "MARS")

#####################eXtreme GB##########
library(xgboost)
library(mlr)
## 調參
train.task <- makeRegrTask(data = train_jp, target = target_jp)

lrn.xgb <- makeLearner("regr.xgboost", predict.type = "response")

lrn.xgb$par.vals <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eval_metric = "rmse",
  nrounds = 100
)

params <- makeParamSet(
  makeIntegerParam("max_depth", lower = 3, upper = 8),
  makeNumericParam("eta", lower = 0.01, upper = 0.05),
  makeNumericParam("gamma", lower = 0.05, upper = 0.1),
  makeNumericParam("lambda", lower = 1, upper = 10),
  makeNumericParam("subsample", lower = 0.5, upper = 1),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 1)
)

print("Start Tuning Parameters...")
mytune <- tuneParams(
  learner = lrn.xgb, 
  task = train.task, 
  measures = rmse, 
  par.set = params, 
  show.info = TRUE,
  resampling = makeResampleDesc("CV", iters = 3), # 3-fold CV
  control = makeTuneControlRandom(maxit = 10)     # RandomSearch with 10 iterations
)

print("Best Parameters Found:")
print(mytune$x)

## 建模訓練
lrn_tune <- setHyperPars(lrn.xgb, par.vals = mytune$x)
xgb_jp <- mlr::train(learner = lrn_tune, task = train.task)

## 預測
pred.xgb_jp <- predict(xgb_jp, newdata = test_jp)
tableJP[1,1] <- mean((pred.xgb_jp$data$truth - pred.xgb_jp$data$response)^2)^0.5 #RMSE
tableJP[2,1] <- mean(abs(pred.xgb_jp$data$truth-pred.xgb_jp$data$response)) #MAE                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        .xgb$data$response)) #MAE
tableJP[3,1] <- mean(abs((pred.xgb_jp$data$truth - pred.xgb_jp$data$response)/pred.xgb_jp$data$truth)) #MAPE

#####################GB##########
library(caret) #調參用
library(gbm)
## Tunning Parameters調參
numfolds_gb = trainControl(method = "cv", number=3)
grid_gb = expand.grid( n.trees = seq(80,160,20), interaction.depth=seq(3,8), shrinkage = c(0.01,0.05,0.1), n.minobsinnode = c(7,9,11,13,15))
cv_gb = caret::train(Japan~., data = train_jp, method="gbm", trControl=numfolds_gb, tuneGrid=grid_gb, train.fraction=0.6)
cv_gb$bestTune

## 建模訓練
gb_jp<-gbm(Japan~., data =train_jp, n.trees= cv_gb$bestTune[,1], 
           interaction.depth=cv_gb$bestTune[,2], 
           shrinkage=cv_gb$bestTune[,3], n.minobsinnode=cv_gb$bestTune[,4])
summary(gb_jp)

## 預測
pred.gb_jp <- predict(gb_jp, newdata = test_jp, n.trees = cv_gb$bestTune[,1])

tableJP[1,2] <- mean((test_jp$Japan-pred.gb_jp)^2)^0.5 #RMSE
tableJP[2,2] <- mean(abs(test_jp$Japan-pred.gb_jp)) #MAE
tableJP[3,2] <- mean(abs((test_jp$Japan-pred.gb_jp)/test_jp$Japan)) #MAPE

#####################MARS##########
library(earth)
# A.建立 MARS 模型
# degree=2 允許變數間的交互作用 (Interaction)
set.seed(42)
mars_jp <- earth(x = train_jp[,-11], 
                 y = train_jp$Japan, 
                 degree = 2)

# B.查看 MARS 模型摘要與變數重要性
summary(mars_jp)
mars_imp.jp <- evimp(mars_jp) # Estimate Variable IMPortance
print("MARS Variable Importance:")
print(mars_imp.jp)

# C.預測
# 注意：earth 的 predict 會回傳矩陣，必須用 [,1] 轉成向量
pred.mars_jp <- predict(mars_jp, newdata = test_jp)[, 1]

tableJP[1, 3] <- mean((test_jp$Japan - pred.mars_jp)^2)^0.5 # RMSE
tableJP[2, 3] <- mean(abs(test_jp$Japan - pred.mars_jp)) # MAE
tableJP[3, 3] <- mean(abs((test_jp$Japan - pred.mars_jp) / test_jp$Japan)) # MAPE

# 顯示最終完整表格
print(tableJP)

#####繪製預測JP的綜合比較圖##########################
## 1. 對「全時段 (2011-2025)」進行預測
# 為了畫出連貫的圖，需要對整個 df 進行預測
df_full <- Asia_index %>%
  select(Date, all_of(target_var), all_of(union_features))

# (A) XGB 預測 (只使用KPI變數，因為XGB模型不會自己抓)
pred_xgb_all <- predict(xgb_jp, newdata = df_full[, union_features])$data$response
# (B) GB 預測 (使用 gbm)
pred_gb_all <- predict(gb_jp, newdata = df_full, n.trees = cv_gb$bestTune[,1])
# (C) MARS 預測 (使用 earth)
# 記得加 [,1] 轉成向量
pred_mars_all <- predict(mars_jp, newdata = df_full)[, 1]

## 2. 整合繪圖資料
library(tidyverse)

# 建立一個專門畫圖的資料框
plot_df <- df_full %>%
  select(Date, Japan) %>%  # 選出時間與真實值
  mutate(
    XGB = pred_xgb_all,
    GB = pred_gb_all,
    MARS = pred_mars_all
  )

# 轉換成長格式 (Long Format) 以便 ggplot 自動產生圖例
plot_df_long <- plot_df %>%
  pivot_longer(
    cols = c("Japan", "XGB", "GB", "MARS"),
    names_to = "Model",
    values_to = "Index_Value"
  )


## 3. 繪製圖表 (ggplot2)
library(ggplot2)
ggplot(plot_df_long, aes(x = Date, y = Index_Value, color = Model, linetype = Model)) +
  # 1. 畫線
  geom_line(size = 0.8, alpha = 0.8) +
  
  # 2. 設定顏色 (真實值黑色，其他彩色)
  scale_color_manual(values = c("Japan" = "black", "XGB" = "red", "GB" = "blue", "MARS" = "green")) +
  
  # 3. 設定線條型式 (真實值實線，預測虛線)
  scale_linetype_manual(values = c("Japan" = "solid", "XGB" = "longdash", "GB" = "longdash", "MARS" = "longdash")) +
  
  # 4. 加入訓練/測試分隔線 (2022-01-01)
  geom_vline(xintercept = as.numeric(as.Date("2022-01-01")), color = "grey40", linetype = "dashed", size = 1) +
  
  # 5. 加入文字標籤 (Training vs Testing)
  annotate("text", x = as.Date("2016-01-01"), y = max(df_full$Japan), label = "Training (2011-2021)", size = 5, fontface = "bold") +
  annotate("text", x = as.Date("2023-12-01"), y = max(df_full$Japan), label = "Testing (2022-2025)", size = 5, fontface = "bold") +
  
  # 6. 標題與版面美化
  labs(title = "Japan Index Prediction: Actual vs Predictive Models",
       subtitle = "Comparing XGB, GB, and MARS Performance across Training and Testing Periods",
       x = "Year", y = "Index Value") +
  theme_minimal() +
  theme(legend.position = "bottom", 
        plot.title = element_text(face = "bold", size = 16))


##########測試_加上前三KPI的圖######
# 建立一個專門畫圖的資料框
plot_df <- df_full %>%
  select(Date, Japan,DJI,gold,SP500,SOX) %>%  # 選出時間與真實值
  mutate(
    MARS = pred_mars_all
  )

# 轉換成長格式 (Long Format) 以便 ggplot 自動產生圖例
plot_df_long <- plot_df %>%
  pivot_longer(
    cols = c("Japan", "DJI","gold","SP500","SOX", "MARS"),
    names_to = "Model",
    values_to = "Index_Value"
  )

ggplot(plot_df_long, aes(x = Date, y = Index_Value, color = Model, linetype = Model)) +
  # 1. 畫線
  geom_line(size = 0.8, alpha = 0.8) +
  
  # 2. 設定顏色 (真實值黑色，其他彩色)
  scale_color_manual(values = c("Japan" = "black", "DJI" = "red", "gold" = "blue", "SP500" = "green", "SOX" ="purple","MARS" ="orange")) +
  
  # 3. 設定線條型式 (真實值實線，預測虛線)
  scale_linetype_manual(values = c("Japan" = "solid", "DJI" = "longdash", "gold" = "longdash", "SP500" = "longdash", "SOX" = "longdash", "MARS" = "longdash")) +
  
  # 4. 加入訓練/測試分隔線 (2022-01-01)
  geom_vline(xintercept = as.numeric(as.Date("2022-01-01")), color = "grey40", linetype = "dashed", size = 1) +
  
  # 5. 加入文字標籤 (Training vs Testing)
  annotate("text", x = as.Date("2016-01-01"), y = max(df_full$Japan), label = "Training (2011-2021)", size = 5, fontface = "bold") +
  annotate("text", x = as.Date("2023-06-01"), y = max(df_full$Japan), label = "Testing (2022-2025)", size = 5, fontface = "bold") +
  
  # 6. 標題與版面美化
  labs(title = "Japan Index Prediction: Actual vs MARS Prediction",
       x = "Year", y = "Index Value") +
  theme_minimal() +
  theme(legend.position = "bottom", 
        plot.title = element_text(face = "bold", size = 16))

##########測試_加上knot的圖######
library(tidyverse)
library(lubridate)

# =======================================================
# 1. 準備資料 (沿用您的步驟)
# =======================================================
# 假設 Asia_index 已經讀入且 mars_jp 已經訓練好
df_full <- Asia_index %>%
  select(Date, all_of(target_var), all_of(union_features))

# 產生 MARS 預測值
pred_mars_all <- predict(mars_jp, newdata = df_full)[, 1]

# 建立畫圖資料框
plot_df <- df_full %>%
  select(Date, Japan, DJI, gold, SP500, SOX) %>%
  mutate(MARS = pred_mars_all)

# 轉成長格式
plot_df_long <- plot_df %>%
  pivot_longer(
    cols = c("Japan", "DJI", "gold", "SP500", "SOX", "MARS"),
    names_to = "Model",
    values_to = "Index_Value"
  )

# =======================================================
# 2. 關鍵步驟：找出變數穿越 Knot 的「交叉點」
# =======================================================
# 定義您找到的 Knots
knot_values <- list(
  "DJI" = 31496.3,
  "SP500" = 2105.2,
  "gold" = 1268.1
)

# 寫一個小函數來抓出「穿越瞬間」的日期
find_knot_points <- function(data, var_name, threshold) {
  data %>%
    select(Date, Value = all_of(var_name)) %>%
    arrange(Date) %>%
    mutate(
      # 判斷是否穿越：(現在的值 - Knot) * (前一天的值 - Knot) < 0 代表正負號改變，即穿越
      cross = (Value - threshold) * lag(Value - threshold) < 0
    ) %>%
    filter(cross == TRUE) %>%
    mutate(
      Model = var_name,          # 標記是哪個變數
      Index_Value = threshold,   # Y軸位置定在 Knot 值上 (畫在線上)
      Label = paste0(var_name, " Knot\n", threshold) # 標籤文字
    ) %>%
    select(Date, Model, Index_Value, Label)
}

# 抓出三個變數的穿越點
knot_points_df <- bind_rows(
  find_knot_points(df_full, "DJI", knot_values$DJI),
  find_knot_points(df_full, "SP500", knot_values$SP500),
  find_knot_points(df_full, "gold", knot_values$gold)
)

# =======================================================
# 3. 繪圖 (加入 Knot 標記)
# =======================================================
ggplot(plot_df_long, aes(x = Date, y = Index_Value, color = Model, linetype = Model)) +
  # 1. 畫線 (主圖)
  geom_line(size = 0.8, alpha = 0.8) +
  
  # 2. 加入 Knot 水平線 (淡淡的虛線，幫助對齊)
  geom_hline(yintercept = knot_values$DJI, color = "red", linetype = "dotted", alpha = 0.4) +
  geom_hline(yintercept = knot_values$SP500, color = "green", linetype = "dotted", alpha = 0.4) +
  geom_hline(yintercept = knot_values$gold, color = "blue", linetype = "dotted", alpha = 0.4) +
  
  # 3. 加入 Knot 交叉點 (打叉叉標記)
  geom_point(data = knot_points_df, aes(x = Date, y = Index_Value, color = Model), 
             shape = 4, size = 3, stroke = 1.5, show.legend = FALSE) +
  
  # 4. (選用) 加入 Knot 文字標籤 (只標示第一次穿越，避免太亂)
  # 這裡只標出每個變數「最後一次」穿越的時間點，比較好讀
  geom_text(data = knot_points_df %>% group_by(Model) %>% slice_tail(n = 1), 
            aes(label = paste(Model, "Knot:", Index_Value)), 
            vjust = -1, size = 3, show.legend = FALSE) +
  
  # 5. 設定顏色與線條 (您原本的設定)
  scale_color_manual(values = c("Japan" = "black", "DJI" = "red", "gold" = "blue", 
                                "SP500" = "green", "SOX" = "purple", "MARS" = "orange")) + 
  scale_linetype_manual(values = c("Japan" = "solid", "DJI" = "longdash", "gold" = "longdash", 
                                   "SP500" = "longdash", "SOX" = "longdash", "MARS" = "solid")) +
  
  # 6. 分隔線與標註
  geom_vline(xintercept = as.numeric(as.Date("2022-01-01")), color = "grey40", linetype = "dashed", size = 1) +
  annotate("text", x = as.Date("2016-01-01"), y = max(df_full$Japan, na.rm=T), label = "Training", size = 5, fontface = "bold") +
  annotate("text", x = as.Date("2023-06-01"), y = max(df_full$Japan, na.rm=T), label = "Testing", size = 5, fontface = "bold") +
  
  # 7. 版面美化
  labs(title = "Impact of Knots on MARS Prediction",
       subtitle = "X marks indicate where variables cross their MARS knots",
       x = "Year", y = "Index Value") +
  theme_minimal() +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold", size = 16))