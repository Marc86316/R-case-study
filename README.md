# R-case-study

## 1. 金融與經濟分析 (Finance & Economics)

包含總體經濟指標預測、銀行行銷活動分析與投資組合最佳化。

| 資料集 (Dataset) | 內容 | 技術線 (Techniques & Models) |
| :--- | :--- | :--- |
| **Asia_index** | 預測亞洲國家（台、日、韓、印）的指數；計算各國風險與報酬並比較；尋找關鍵績效指標 (KPI) | SMA、指數平滑、MLR、MARS、KNN迴歸、F檢定、T檢定、XGBoost、Gradient Boosting (GB) |
| **bank** | 找出客戶是否貸款/定存的關聯規則；分析不同年齡、婚姻狀態與帳戶餘額、貸款的關聯；以分類器預測客戶行為 | 資料離散化、關聯規則、Logit、Naive Bayes、T檢定、卡方檢定、比例檢定、ANOVA、Random Forest、AdaBoost、GB、XGBoost、Bagging |
| **ETF (00850, 0051)** | 挑選成分股並下載股價，利用演算法推導出最佳投資組合（最大化報酬、最小化風險等策略） | 數學規劃 (Mathematical programming)、非線性規劃、強化學習 (Reinforcement Learning)、投資組合優化 |
| **usconsumption** | 美國消費趨勢之時間序列分析與未來消費量預測 | 時間序列、單根檢定 (ADF/PP/KPSS)、ARIMA |

## 2. 製造與科技業 (Manufacturing & Tech)

涵蓋消費性電子產品銷量、半導體晶片設計及工業設備壽命預測。

| 資料集 (Dataset) | 內容 | 技術線 (Techniques & Models) |
| :--- | :--- | :--- |
| **electronics** | 對台積電、蘋果及各大廠的晶片/消費性電子產品銷售總額進行預測，並尋找最佳的時間落後 (time lags) 設定 | ARIMA、Holt-Winters、VAR、動態 ARIMA、lag-ARIMA、SVR、ANN (MLP)、MLR、MARS、SARIMA |
| **IC_design** | 半導體 IC 設計公司的 EPS 與 ROE 預測，並透過降維技術處理高維度特徵 | PCA、最大變異法轉軸 (Varimax)、KNN、MARS、MLR、Sliced Inverse Regression (SIR) |
| **MB_revenue** | 電腦主機板、桌機、筆電與伺服器出貨量之進階時間序列與關聯預測 | Holt-Winters、SARIMA、動態 ARIMA、落後模型 (Lag)、VAR |
| **NASA FD001** (渦輪引擎) | 特徵工程篩選感測器訊號，預測渦輪引擎之剩餘使用壽命 (RUL) | PCA、Mutual Information (互資訊)、Random Forest、XGBoost、DNN、GRU、SVR |

## 3. 能源與原物料 (Energy & Commodities)

著重於電力需求預測、發電廠容量最佳化及原物料價格預測。

| 資料集 (Dataset) | 題目內容 | 技術線 (Techniques & Models) |
| :--- | :--- | :--- |
| **electricity / electricity_optimization** | 進行電力需求預測；針對成本與碳排放政策進行動態最佳化與模擬分析；使用強化學習尋求平衡情境最佳解 | 動態最佳化、模擬 (Simulation)、線性規劃、強化學習、Holt-Winters、SARIMA、MLR、MARS、VAR |
| **material** | 針對油、煤、氣及金、銀、銅等原物料建構時間序列與迴歸預測模型，並篩選 KPI | SARIMA、VAR、MARS、SVR |
| **風機資料集** | 處理不平衡資料（正常/異常樣本），選擇 KPI，並比較不同分類器判斷風機故障之績效 | SMOTE、RUS、SFS、Random Forest、XGBoost、DNN、CNN、SVC |

## 4. 客戶關係與社會科學 (CRM & Social Science)

包含電信客戶流失分析與民意調查統計。

| 資料集 (Dataset) | 內容 | 技術線 (Techniques & Models) |
| :--- | :--- | :--- |
| **TelecomChurn (churn)** | 分析電信客戶流失特徵；篩選共同預測因子；比較決策樹與集成學習在交叉驗證上的 AUC | CART、Random Forest、Gradient Boosting、C5.0、AdaBoost (Adabag)、ROC/AUC 曲線 |
| **public opinion polling** | 民眾對政府能源政策（核能/綠能/煤氣）的偏好分佈與性別是否有差異 | 卡方檢定 (Chi-square test) |

## 5. 運動分析 (Sports Analytics)

聚焦於職業運動賽事之球員表現數據分群、打擊指標降維與進階指標探索。

| 資料集 (Dataset) | 內容 | 技術線 (Techniques & Models) |
| :--- | :--- | :--- |
| **2011NBA** | 針對 NBA 球員表現數據進行集群分析，探討球員角色分群，並以 ANOVA 分析集群與得分表現是否相關 | 階層式分群 (Ward-distance)、One-way ANOVA |
| **2012MLB** | 使用美國職棒大聯盟的各項打擊數據（如攻擊指標等）進行降維萃取並視覺化評估 | PCA、陡坡圖 (Scree Plot)、因素負荷量圖 (Biplot/Dotchart) |

## 6. 醫療與健康照護 (Medical & Healthcare)

此分類著重於病患資料分析、疾病預測與身體特徵之離群值與分類探討。

| 資料集 (Dataset) | 內容 | 技術線 (Techniques & Models) |
| :--- | :--- | :--- |
| **pima** | 糖尿病離群值檢測、特徵分群、關聯規則探討；建立分類模型預測是否罹患糖尿病；檢定健康與患病群體的顯著差異 | Mahalanobis距離、Isolation Forest、Fuzzy C-means、關聯規則(Apriori)、Logit、KNN、T檢定、F檢定、SVM、CART、BPN-MLP |
| **gender_outlier / gender_size** | 探索男女身高體重腰圍之分佈與離群值；使用MDS降維；對性別進行分群與分類預測；男女特徵差異的統計檢定 | 直方圖、盒狀圖、MDS、階層式分群(DB index)、Mahalanobis、Isolation Forest、Gaussian Mixture (EM)、KNN、Naive Bayes、SVM、T檢定、ANOVA |
| **breast_cancer** | 預測乳癌為良性或惡性，並評估不同演算法在交叉驗證下的分類績效 | KNN、Naive Bayes、Logistic Regression、ANN (neuralnet)、3-fold Cross Validation |

## 7. 物流與航運管理 (Logistics & Transportation)

聚焦於航運運量預測、運費與各型船舶（貨櫃、輕便型、海岬型）KPI探索。

| 資料集 (Dataset) | 內容 | 技術線 (Techniques & Models) |
| :--- | :--- | :--- |
| **container** | 訓練模型以預測貨櫃營收；尋找重要預測變數並解釋因果關係；對進階迴歸器進行交叉驗證績效比較 | 偏誤迴歸 (Ridge, Lasso, ElasticNet)、MLR、SVR、GB、MARS、CART、Random Forest、XGBoost、KNN、Cross Validation |
| **handysize** | 針對輕便型散裝船，辨識關鍵指標並預測市場數據 | MLR、MARS、KNN、Random Forest、XGBoost |
| **capesize** | 海岬型散裝船資料集的簡單迴歸建模與績效評估 | MLR、MARS、KNN迴歸 |
| **AirPassengers** | 航空客運量的季節性分析與預測 | 時間序列差分、自動化 ARIMA 建模 |

## 8. 食品與品質管控 (Food & Quality Control)

專注於紅酒與白酒的成分分析、品質評分 (Regression) 與等級分類 (Classification)。

| 資料集 (Dataset) | 內容 | 技術線 (Techniques & Models) |
| :--- | :--- | :--- |
| **winequality-red** | 將品質標籤離散化為分類問題；進行降維並測試分群與分類績效；建構紅酒品質迴歸模型 | K-means、K-medoids、PCA、Logit、Naive Bayes、SVM、KNN、MARS、CART、Random Forest、AdaBoost、GB、ANN (nnet) |
| **winequality-white** | 以PCA萃取特徵並分類；執行5-fold cross validation進行白酒品質預測與分類模型比較 | PCA、SVM、ANN (neuralnet, nnet)、XGBoost、SVR、AdaBoost、Gradient Boosting、Random Forest、Bagging |

## 9. 其他經典資料集 (Classic Datasets)

常用於演算法展示、基礎統計或探索性資料分析 (EDA) 的經典開源資料。

| 資料集 (Dataset) | 內容 | 技術線 (Techniques & Models) |
| :--- | :--- | :--- |
| **iris** | 探索性資料分析、常態性檢定；測試各種分群演算法並視覺化；多類別分類與 PCA 測試 | 直方圖、相關係數、K-means、K-medoids、Fuzzy C-means、階層式分群、Gaussian Mixture、KNN、Naive Bayes、多類別 Logit、常態檢定(Shapiro/QQ-plot)、Random forest、PCA、SVM |
| **boston** | 使用多種線性與偏誤迴歸模型預測波士頓房價並進行誤差比較 | MLR、Ridge、Lasso、ElasticNet |
| **titanic** | 預測鐵達尼號乘客生存率，探討性別、年齡、艙等對生存機率的影響並比較模型與 ROC | Naive Bayes、Logit 模型、C5.0 決策樹、CART、ROC/AUC 評估 |
