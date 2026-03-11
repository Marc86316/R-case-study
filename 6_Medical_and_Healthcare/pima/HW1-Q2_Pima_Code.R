##HW#2 根據Pima(印地安人糖尿病)資料集，用Mahalanobis distance+Chi-square test，找出10%的離群值
Pima.all=read.csv("Pima.csv", header=TRUE, sep=",")
Pima.num <- Pima.all[, 1:8]  # 前 8 個都是連續變數
pima.mean <- colMeans(Pima.num) #計算每一欄數值的平均數(找出中心點)
pima.cov  <- cov(Pima.num, use = "pairwise") #計算共變異矩陣，看各變數之間有沒有一起變動的關係，若有少數缺值，直接用有的數值去算
pima.md <- mahalanobis(Pima.num, pima.mean, pima.cov) #計算mahalanobis distance

#用 Chi-square 找 10% 門檻
cutoff.md <- qchisq(p = 0.90, df = ncol(Pima.num))
cutoff.md
md.outlier <- pima.md > cutoff.md #標記 Mahalanobis outliers
mean(md.outlier) #確認Outlier占整體資料比例是多少

md.index <- which(md.outlier == TRUE) #算有哪幾筆資料是離群值
md.index
length(md.index) #算有幾筆資料是離群值

##使用Isolation Forest進行離群值偵測
install.packages("isotree")
library(isotree)
iso.model <- isolation.forest(
  Pima.num, #跟 Mahalanobis 用同一份資料
  ntrees = 100, #多切幾次，結果比較穩定
  seed = 123 #固定亂數
)
iso.score <- predict(iso.model, Pima.num) #每一筆資料一個「異常分數」
summary(iso.score)

cutoff.iso <- quantile(iso.score, 0.90) #算90%的分數門檻
cutoff.iso
iso.outlier <- iso.score > cutoff.iso #標記哪些是離群值
mean(iso.outlier) #確認Outlier占整體資料比例是多少

iso.index <- which(iso.outlier == TRUE) #算有哪幾筆資料是離群值
iso.index
length(iso.index) #算有幾筆資料是離群值

##找共同的離群值（common outliers）
common.outliers <- intersect(md.index, iso.index)
common.outliers
length(common.outliers)

##只有 Mahalanobis 抓到的
md.only <- setdiff(md.index, iso.index)
md.only
length(md.only)
##只有 Isolation Forest 抓到的
iso.only <- setdiff(iso.index, md.index)
iso.only
length(iso.only)

##表格化
#把結果加回 Pima.all
Pima.result <- Pima.all
Pima.result$Mahalanobis <- md.outlier
Pima.result$IsolationForest <- iso.outlier

#新增一欄「離群值類型標記」，每一筆資料都被清楚分類為四種之一(暫且不用)
Pima.result$OutlierType <- "Normal"

Pima.result$OutlierType[
  Pima.result$Mahalanobis & Pima.result$IsolationForest
] <- "Common outlier"

Pima.result$OutlierType[
  Pima.result$Mahalanobis & !Pima.result$IsolationForest
] <- "Mahalanobis only"

Pima.result$OutlierType[
  !Pima.result$Mahalanobis & Pima.result$IsolationForest
] <- "Isolation Forest only"

#三種 outlier 各自篩選成表格
Mahalanobis_only_table <- Pima.result[
  Pima.result$Mahalanobis & !Pima.result$IsolationForest,
]

IsolationForest_only_table <- Pima.result[
  !Pima.result$Mahalanobis & Pima.result$IsolationForest,
]

Common_outlier_table <- Pima.result[
  Pima.result$Mahalanobis & Pima.result$IsolationForest,
]

#快速確認每張表各有幾筆（可留可不留）
nrow(Mahalanobis_only_table)
nrow(IsolationForest_only_table)
nrow(Common_outlier_table)

#切掉Mahalanobis和IsolationForest欄位
Mahalanobis_only_table <- Mahalanobis_only_table[, names(Pima.all)]
IsolationForest_only_table <- IsolationForest_only_table[, names(Pima.all)]
Common_outlier_table <- Common_outlier_table[, names(Pima.all)]


--------------------------------------------------------
###outlier test based on Mahalanobis distance
gender_outlier <-read.csv("gender_outlier.csv")
attach(gender_outlier)
Male=subset(gender_outlier, Gender=="male")
Female=subset(gender_outlier, Gender=="female")

Ma.mean<-colMeans(Male[, -4])
Ma.mean
Fe.mean<-colMeans(Female[, -4])
Fe.mean

Ma.var<-cov(Male[c(1:3)], use="pairwise")
Ma.var
Fe.var<-cov(Female[c(1:3)], use="pairwise")
Fe.var

Male$mdis<- mahalanobis(Male[, -4], Ma.mean, Ma.var)
Female$mdis<-mahalanobis(Female[, -4], Fe.mean, Fe.var)

Male$maout<-(Male$mdis> qchisq(df=3, p=0.95)) #Chi-square
Female$feout<-(Female$mdis> qchisq(df=3, p=0.95)) #df: degree of freedom

attach(Male)
which(maout==TRUE)

attach(Female)
which(feout==TRUE)

###outlier test based on isolation forest
#install.packages("isotree")
library(isotree)
library(dplyr)

iso_m <- isolation.forest(Male[,-4], ntrees = 100, nthreads = -1)
pred_m <- predict(iso_m, Male[,-4])
Male[which.max(pred_m),] #Data with highest outlier score
Male[which(pred_m>0.6),]

iso_f <- isolation.forest(Female[,-4], ntrees = 100, nthreads = -1)
pred_f <- predict(iso_f, Female[,-4])
Female[which.max(pred_f),] #Data with highest outlier score
Female[which(pred_f>0.6),]