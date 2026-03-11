library(quantmod)
library(dplyr)
library(lubridate)
library(nloptr)
library(ggplot2)
library(tidyr)

# ===============================
# 1. 資料設定與下載
# ===============================
stocks <- c("2330.TW", "2317.TW", "2454.TW", "3231.TW", "2357.TW",
            "2379.TW", "2327.TW", "2881.TW", "2891.TW", "2885.TW",
            "2890.TW", "1102.TW", "2618.TW", "1216.TW", "3045.TW")

# 下載區間：配合第一期測試 (2018/01起)，訓練集需要往前推兩年至 2016/01
start_date <- as.Date("2016-01-01")
end_date   <- as.Date("2026-02-01") # 抓到2月1日確保有1/31的資料

# 下載資料函數
get_stock_data <- function(ticker) {
  tryCatch({
    data <- getSymbols(ticker, src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)
    return(Ad(data)) # 使用調整後收盤價
  }, error = function(e) return(NULL))
}

# 合併資料
price_list <- lapply(stocks, get_stock_data)
price_matrix <- do.call(merge, price_list)
names(price_matrix) <- stocks
price_matrix <- na.omit(price_matrix) # 移除缺失值

# 轉為 Data Frame 並計算日報酬率
data_prices <- data.frame(date = index(price_matrix), coredata(price_matrix),check.names = FALSE)
stock_cols <- 2:ncol(data_prices)
data_returns <- data_prices
data_returns[, stock_cols] <- (data_prices[, stock_cols] / lag(data_prices[, stock_cols]) - 1)
data_returns <- na.omit(data_returns)

# ===============================
# 2. 定義數學規劃優化函數 (Mathematical Programming)
# ===============================
# 通用優化器
run_optimization <- function(train_ret, type, threshold1=NULL, threshold2=NULL) {
  n_assets <- ncol(train_ret)
  # 計算這個訓練集(2年)實際上有幾個交易日
  actual_days <- nrow(train_ret) 
  # 因為訓練集是 2 年，所以平均一年的交易日就是 actual_days / 2
  annual_factor <- actual_days / 2 
  mu <- (1+colMeans(train_ret))** annual_factor-1 # 年化報酬
  sigma <- cov(train_ret) * annual_factor # 年化共變異數
  
  # 目標函數與限制條件
  if (type == "max_return") 
    { # ➊ Maximize Return => Minimize -Return
     eval_f <- function(w) {
      risk <- sqrt(t(w) %*% sigma %*% w)
      ret <- sum(w * mu)
      return(-ret)
    }
    # Constraint(不等式限制): Risk < threshold1 (0.3)
    eval_g_ineq <- function(w) return(sqrt(t(w) %*% sigma %*% w) - threshold1)
    } 
  else if (type == "min_risk") 
    { # ➋ Minimize Risk
     eval_f <- function(w) {
      risk <- sqrt(t(w) %*% sigma %*% w)
      ret <- sum(w * mu)
      return(risk)
    }
    # Constraint(不等式限制): Return > threshold1 (0.1) => 0.1 - Return < 0
    eval_g_ineq <- function(w) return(threshold2 - sum(w * mu))
  } 
  else if (type == "trade_off") 
    { # ➌ Maximize (Return - Risk) => Minimize -(Return - Risk)
     eval_f <- function(w) {
      risk <- sqrt(t(w) %*% sigma %*% w)
      ret <- sum(w * mu)
      return(-(ret - risk))
    }
    # Constraints限制: Risk < 0.25 AND Return > 0.15
    eval_g_ineq <- function(w) {
      risk <- sqrt(t(w) %*% sigma %*% w)
      ret <- sum(w * mu)
      return(c(risk - threshold1, threshold2 - ret)) # risk < th1, ret > th2
    }
  }
  
  # 權重總和 = 1(等式限制)
  eval_g_eq <- function(w) return(sum(w) - 1)
  
  # 執行優化
  opts <- list("algorithm" = "NLOPT_LN_COBYLA", "xtol_rel" = 1e-6, "maxeval" = 5000)
  res <- nloptr(x0 = rep(1/n_assets, n_assets), eval_f = eval_f, 
                eval_g_ineq = eval_g_ineq, eval_g_eq = eval_g_eq, 
                lb = rep(0, n_assets), ub = rep(1, n_assets), opts = opts)
  return(res$solution)
}

# ===============================
# 3. 滾動視窗回測 (Rolling Window Backtest)
# ===============================
# 產生每半年的測試起點 (2018-01-01 到 2025-07-01)
test_starts <- seq(as.Date("2018-01-01"), as.Date("2025-07-01"), by = "6 months")

# 計算總投入本金 (為了與 DCA 公平比較)
# DCA 每月 10000，半年 6萬。總共有 length(test_starts) 個半年。
money=10000
total_months <- length(test_starts) * 6
initial_wealth <- total_months * money  # 單筆策略一開始就投入這筆總金額

results_wealth <- data.frame(Date = test_starts, 
                  Max_Return = 0, Min_Risk = 0, Trade_Off = 0, Equal_Weight = 0, DCA = 0)

# 策略當前淨值初始化
current_wealth <- c(Max_Return = initial_wealth, Min_Risk = initial_wealth, 
                    Trade_Off = initial_wealth, Equal_Weight = initial_wealth)

# DCA 專用變數
dca_shares <- rep(0, length(stocks)) 
dca_cumulative_cost <- 0 

#記錄三種策略的權重
weights_history_max <- data.frame()
weights_history_min <- data.frame()
weights_history_trade <- data.frame()

cat("開始執行【半年期】滾動回測 (2018~2025)...\n")

for (i in 1:length(test_starts)) {
  # 1. 定義時間視窗
  test_start_date <- test_starts[i]
  # 測試集：接下來的 6 個月
  test_end_date <- test_start_date + months(6) - days(1) 
  
  # 訓練集：往前推 2 年 (4 個半年)
  train_end_date <- test_start_date - days(1)
  train_start_date <- train_end_date - years(2) + days(1)
  
  cat(sprintf("第 %2d 期 | 訓練: %s ~ %s | 測試: %s ~ %s \n", 
              i, train_start_date, train_end_date, test_start_date, test_end_date))
  
  # 2. 切割資料
  train_set <- data_returns %>% filter(date >= train_start_date & date <= train_end_date)
  train_matrix <- train_set[, stock_cols]
  
  test_set <- data_returns %>% filter(date >= test_start_date & date <= test_end_date)
  test_matrix <- test_set[, stock_cols]
  
  # 3. 計算該「半年」各策略的最佳權重
  w_max_ret <- tryCatch(run_optimization(train_matrix, "max_return", threshold1=0.3), error=function(e) rep(1/15,15))
  w_min_risk <- tryCatch(run_optimization(train_matrix, "min_risk", threshold1=0.12), error=function(e) rep(1/15,15))
  w_trade_off <- tryCatch(run_optimization(train_matrix, "trade_off", threshold1=0.25, threshold2=0.15), error=function(e) rep(1/15,15))
  w_equal <- rep(1/length(stocks), length(stocks))
  
  # 👉 把算出來的權重存進歷史紀錄表裡面
  tmp_max <- data.frame(Date = test_start_date, t(round(w_max_ret, 4)))
  tmp_min <- data.frame(Date = test_start_date, t(round(w_min_risk, 4)))
  tmp_trade <- data.frame(Date = test_start_date, t(round(w_trade_off, 4)))
  
  colnames(tmp_max)[-1] <- stocks
  colnames(tmp_min)[-1] <- stocks
  colnames(tmp_trade)[-1] <- stocks
  
  weights_history_max <- rbind(weights_history_max, tmp_max)
  weights_history_min <- rbind(weights_history_min, tmp_min)
  weights_history_trade <- rbind(weights_history_trade, tmp_trade)
  
  # 4. 計算這 6 個月的累積報酬率 (Half-year Holding Return)
  # 取測試期第一天與最後一天的價格來計算該半年的總報酬
  price_test_start <- as.numeric(data_prices %>% filter(date >= test_start_date) %>% head(1) %>% select(all_of(stocks)))
  price_test_end <- as.numeric(data_prices %>% filter(date <= test_end_date) %>% tail(1) %>% select(all_of(stocks)))
  stock_semi_annual_ret <- (price_test_end / price_test_start) - 1
  
  # 5. 更新單筆策略淨值
  port_ret_max <- sum(w_max_ret * stock_semi_annual_ret)
  port_ret_min <- sum(w_min_risk * stock_semi_annual_ret)
  port_ret_trade <- sum(w_trade_off * stock_semi_annual_ret)
  port_ret_eq <- sum(w_equal * stock_semi_annual_ret)
  
  current_wealth["Max_Return"] <- current_wealth["Max_Return"] * (1 + port_ret_max)
  current_wealth["Min_Risk"] <- current_wealth["Min_Risk"] * (1 + port_ret_min)
  current_wealth["Trade_Off"] <- current_wealth["Trade_Off"] * (1 + port_ret_trade)
  current_wealth["Equal_Weight"] <- current_wealth["Equal_Weight"] * (1 + port_ret_eq)
  
  # 6. 處理 DCA (這半年內，每個月初扣款 10000)
  # 抓出這半年內的每個月第一天
  months_in_semester <- seq(test_start_date, test_end_date, by = "month")
  for (m_date in months_in_semester) {
    # 找到當月第一個交易日的價格
    p_month_start <- as.numeric(data_prices %>% filter(date >= m_date) %>% head(1) %>% select(all_of(stocks)))
    # 買入股數 (10000元均分給15檔)
    #shares_bought <- (money / length(stocks)) / p_month_start
    dca_shares <- dca_shares + (money / length(stocks)) / p_month_start
    dca_cumulative_cost <- dca_cumulative_cost + money
  }
  
  # 計算 DCA 在這半年期末的總市值
  dca_market_value <- sum(dca_shares * price_test_end)
  
  # 7. 儲存本期末結果
  results_wealth[i, "Max_Return"] <- current_wealth["Max_Return"]
  results_wealth[i, "Min_Risk"] <- current_wealth["Min_Risk"]
  results_wealth[i, "Trade_Off"] <- current_wealth["Trade_Off"]
  results_wealth[i, "Equal_Weight"] <- current_wealth["Equal_Weight"]
  results_wealth[i, "DCA"] <- dca_market_value
}

cat("\n回測完成！\n")

# ===============================
# 4. 結果視覺化與分析
# ===============================
print(results_wealth)

# 為了畫圖，將資料轉長格式
df_long <- pivot_longer(results_wealth, cols = -Date, names_to = "Strategy", values_to = "Wealth")

# 繪製財富累積圖
ggplot(df_long, aes(x = Date, y = Wealth, color = Strategy)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  labs(title = "Portfolio Performance (2018 - 2025)",
       subtitle = "Rolling Window Optimization (Training: Previous 2 Years, Rebalanced Semi-Annually)",
       y = "Portfolio Value (TWD)", x = "Date") +
  theme_minimal() +
  scale_y_continuous(labels = scales::comma)
  scale_x_date(date_breaks = "1 year", date_labels = "%Y")

# 計算最終績效表
final_row <- tail(results_wealth, 1)
performance_table <- data.frame(
  Strategy = colnames(results_wealth)[-1],
  Final_Value = as.numeric(final_row[-1]),
  Return_Rate = c(
    (final_row$Max_Return - initial_wealth) / initial_wealth,
    (final_row$Min_Risk - initial_wealth) / initial_wealth,
    (final_row$Trade_Off - initial_wealth) / initial_wealth,
    (final_row$Equal_Weight - initial_wealth) / initial_wealth,
    (final_row$DCA - initial_wealth) / initial_wealth
  )
)
performance_table$Return_Percent <- paste0(round(performance_table$Return_Rate * 100, 2), "%")
print(performance_table)

# ===============================
# 5. 繪製策略權重變化的堆疊長條圖
# ===============================
# 👉 ：印出各策略每個月的權重變化(也可以直接看table)
cat("\n--- Max_Return 策略權重變化 ---\n")
print(weights_history_max)
cat("\n--- Min_Risk 策略權重變化 ---\n")
print(weights_history_min)
cat("\n--- Trade_Off 策略權重變化 ---\n")
print(weights_history_trade)

# 1. 將前面儲存的三個權重資料表，加上「策略名稱」的標籤
df_max <- weights_history_max %>% mutate(Strategy = "Max_Return")
df_min <- weights_history_min %>% mutate(Strategy = "Min_Risk")
df_trade <- weights_history_trade %>% mutate(Strategy = "Trade_Off")

# 2. 合併三個資料表
df_all_weights <- bind_rows(df_max, df_min, df_trade)

# 3. 將資料從「寬格式 (Wide)」轉換為「長格式 (Long)」，這是 ggplot2 畫圖的標準格式
df_long_weights <- df_all_weights %>%
  pivot_longer(cols = -c(Date, Strategy), 
               names_to = "Stock", 
               values_to = "Weight")

# 過濾掉權重趨近於 0 的雜訊 (例如小於 0.1% 的權重就不畫出來，讓圖表更乾淨)
df_long_weights <- df_long_weights %>% filter(Weight > 0.001)

# 4. 繪製堆疊長條圖
p_weights <- ggplot(df_long_weights, aes(x = format(Date, "%Y-%m"), y = Weight, fill = Stock)) +
  geom_bar(stat = "identity", position = "stack", color = "white", linewidth = 0.2) +
  # facet_wrap 可以幫我們把三個策略分成三個獨立的子圖，方便上下比較
  facet_wrap(~ Strategy, ncol = 1) + 
  labs(title = "Portfolio Weights Over Time (2018 - 2025)",
       subtitle = "Dynamic Rebalancing of 15 Stocks from 00850",
       x = "Month",
       y = "Weight Allocation",
       fill = "Stock Ticker") +
  theme_minimal() +
  scale_y_continuous(labels = scales::percent) + # Y軸顯示百分比
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 12, face = "bold"), # 讓策略名稱大一點
    legend.position = "right"
  )

# 顯示圖表
print(p_weights)
print(weights_history_trade)