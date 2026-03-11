#install.packages("MASS")
#install.packages("cluster")
#install.packages("clusterSim")
library(MASS)
library(cluster)
library(clusterSim)

winequality <- read.csv("winequality-red.csv")
#remove the last column (quality) 
winequality.new <- winequality[,-12]
# 標準化
wine_scaled <- scale(winequality.new[,1:11])

#install.packages("NbClust")
library(NbClust)
NbClust(wine_scaled[,1:11], distance="euclidean", min.nc=2, max.nc=9, method="kmeans", index="all")
#↑結果顯示K means建議分7群
# 針對 K-means 畫 Elbow Plot，可以看到在7開始趨於平緩
library(factoextra)
fviz_nbclust(wine_scaled, kmeans, method = "wss") +
  geom_vline(xintercept = 7, linetype = 2) + # 加上一條虛線標示您選擇的 k=7
  labs(title = "Elbow Method (Optimal k for K-means)",
       subtitle = "Elbow point indicates optimal clusters")
# 將 method 改成 "silhouette"(輪廓係數，越高越好)
fviz_nbclust(wine_scaled, kmeans, method = "silhouette")

#### K means
set.seed(123) #固定每次分群的順序
result0= kmeans(wine_scaled[,1:11], center=7)
print(result0)
result0$centers
#↑結果看起來兩個sulfur.dioxide有可能是kpi
table(winequality$quality, result0$cluster)
#↑結果看起來第一群可能是品質較好的(?)
##↓用兩個kpi作為xy軸畫圖
plot(wine_scaled[,6:7], pch=result0$cluster, col=result0$cluster)
points(result0$centers[,6:7], col=1:7, pch=8)

## 畫箱型圖找可能的kpi
library(ggplot2)
library(tidyr)
# 1. 結合原始資料與分群結果
plot_data <- data.frame(winequality.new[,1:11], Cluster = as.factor(result0$cluster))
# 2. 轉成長表格以便一次畫出所有特徵
plot_data_long <- pivot_longer(plot_data, cols = -Cluster, names_to = "Feature", values_to = "Value")
# 3. 畫圖
ggplot(plot_data_long, aes(x = Cluster, y = Value, fill = Cluster)) +
  geom_boxplot() +
  facet_wrap(~Feature, scales = "free") + # 讓每個特徵有獨立的Y軸刻度
  theme_minimal() +
  labs(title = "各特徵在不同集群的分佈差異_K means")

## 畫熱力圖找各群特徵
#準備數據：將 centers 轉為 Data Frame
centers_df <- as.data.frame(result0$centers)
centers_df$Cluster <- as.factor(1:nrow(centers_df)) # 加入群組標籤
# 2. 轉為長表格 (Tidy format)
centers_long <- pivot_longer(centers_df, cols = -Cluster, names_to = "Feature", values_to = "Z_Score")
ggplot(centers_long, aes(x = Feature, y = Cluster, fill = Z_Score)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) + # 設定紅藍配色
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "特徵強度熱力圖_K means", fill = "Z-Score")

## 檢查各群的平均品質分數(檢驗分群的效果)
aggregate(winequality$quality, by=list(result0$cluster), mean)
# 1. 建立一個專門畫圖的資料框
# 從原始的 winequality 抓 quality，從 result_scaled 抓分群結果
plot_data <- data.frame(
  Cluster = as.factor(result0$cluster),
  Quality = winequality$quality  # 注意這裡是用原始的 winequality
)
# 2. 畫箱型圖 (Boxplot)
ggplot(plot_data, aes(x = Cluster, y = Quality, fill = Cluster)) +
  geom_boxplot() +
  labs(
    title = "不同集群的品質分數分佈_K means",
    x = "Cluster (集群)",
    y = "Quality Score (評分)"
  ) +
  theme_minimal()

#### K medoids
set.seed(123)
result1=pam(wine_scaled[,1:11], 7)
print(result1)
result1$medoids
table(winequality$quality, result1$cluster)


## 畫箱型圖找可能的kpi
library(ggplot2)
library(tidyr)
# 1. 結合原始資料與分群結果
plot_data <- data.frame(winequality.new[,1:11], Cluster = as.factor(result1$cluster))
# 2. 轉成長表格以便一次畫出所有特徵
plot_data_long <- pivot_longer(plot_data, cols = -Cluster, names_to = "Feature", values_to = "Value")
# 3. 畫圖
ggplot(plot_data_long, aes(x = Cluster, y = Value, fill = Cluster)) +
  geom_boxplot() +
  facet_wrap(~Feature, scales = "free") + # 讓每個特徵有獨立的Y軸刻度
  theme_minimal() +
  labs(title = "各特徵在不同集群的分佈差異_K medoids")

## 畫熱力圖找各群特徵
#準備數據：將 centers 轉為 Data Frame
medoids_df <- as.data.frame(result1$medoids)
medoids_df$Cluster <- as.factor(1:nrow(medoids_df)) # 加入群組標籤
# 2. 轉為長表格 (Tidy format)
medoids_long <- pivot_longer(medoids_df, cols = -Cluster, names_to = "Feature", values_to = "Z_Score")
ggplot(medoids_long, aes(x = Feature, y = Cluster, fill = Z_Score)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) + # 設定紅藍配色
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "特徵強度熱力圖_K medoids", fill = "Z-Score")

## 檢查各群的平均品質分數(檢驗分群的效果)
aggregate(winequality$quality, by=list(result1$cluster), mean)
# 1. 建立一個專門畫圖的資料框
# 從原始的 winequality 抓 quality，從 result1 抓分群結果
plot_data <- data.frame(
  Cluster = as.factor(result1$cluster),
  Quality = winequality$quality  # 注意這裡是用原始的 winequality
)
# 2. 畫箱型圖 (Boxplot)
ggplot(plot_data, aes(x = Cluster, y = Quality, fill = Cluster)) +
  geom_boxplot() +
  labs(
    title = "不同集群的品質分數分佈_K medoids",
    #subtitle = "Group 2 (Cluster 2) 包含較多高分酒，Group 1 (Cluster 1) 表現最差",
    x = "Cluster (集群)",
    y = "Quality Score (評分)"
  ) +
  theme_minimal()

#### 綜合比較
## K means
# --- 1. 計算各群組的特徵平均值 (這就是 K-means 的 Centroids 真實意義) ---
# 使用原始資料 (winequality) 來算，才會有像樣的物理數值
# 這張表就是你要放在報告裡的 "Centroids / Characteristics"
kmeans_characteristics <- aggregate(winequality[,1:11], by=list(Cluster=result0$cluster), mean)
print("=== K-means 各群組特徵平均值 (Characteristics) ===")
print(kmeans_characteristics)
# --- 2. 計算各群組的平均品質 (Average Quality) ---
kmeans_quality <- aggregate(winequality$quality, by=list(Cluster=result0$cluster), mean)
print("=== K-means 各群組平均品質 (Average Quality) ===")
print(kmeans_quality)

## K medoids
# --- 1. 準備工作 ---
# 確保我們有原始資料 (為了看真實數值) 和 分群結果
# --- 2. 找出質心 (Medoids) 的真實數值 ---
# 我們利用 medoids 的索引 (id.med) 回去原始資料抓數值
medoid_indices <- result1$id.med
medoids_real_values <- winequality[medoid_indices, 1:11] # 只抓特徵
medoids_quality <- winequality[medoid_indices, "quality"] # 抓這三瓶代表酒的品質
# 合併成一張漂亮的表
medoids_table <- data.frame(medoids_real_values, 
                            Medoid_Quality = medoids_quality, 
                            Cluster = 1:7)
print("=== 各群組的代表酒 (Centroids/Medoids) 真實數值 ===")
print(medoids_table)
# --- 3. 計算各群組的平均品質 (Average Quality) ---
avg_quality <- aggregate(winequality$quality, by=list(Cluster=result1$clustering), mean)
print("=== 各群組的平均品質 (Average Quality) ===")
print(avg_quality)
# --- 4. 計算各群組的特徵平均值 (用於描述特徵) ---
# 這可以幫你寫出 "這群酒平均來說比較酸" 這種結論
cluster_means <- aggregate(winequality[,1:11], by=list(Cluster=result1$clustering), mean)
print("=== 各群組的整體特徵平均值 ===")
print(cluster_means)

#### 輪廓係數
## K means
# 計算距離矩陣 (Silhouette 需要這個)
dist_matrix <- dist(wine_scaled)
# 計算 K-means 的輪廓係數
sil_kmeans <- silhouette(result0$cluster, dist_matrix)
kmeans_score <- mean(sil_kmeans[, 3])

## K-medoids
sil_kmedoids <- silhouette(result1$clustering, dist_matrix)
kmedoids_score <- mean(sil_kmedoids[, 3])
cat("K-medoids 平均輪廓係數:", kmedoids_score, "\n")

## 繪製輪廓係數圖
library(factoextra)
fviz_silhouette(sil_kmeans,main = "K-means Silhouette Plot")
fviz_silhouette(sil_kmedoids,main = "K-medoids Silhouette Plot")
