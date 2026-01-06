# 服務於Droplet sensor的 ResNet AI圖象辨識

## 使用ResNet做圖片辨識
1. 使用pytorch作為本次的核心程式庫
2. 微調了ResNet18，用於辨識是否有變化
3. 因為圖片與預訓練的圖片差異極大，所以採取了全微調(Full fine-tuning)策略
4. 看Loss及Accuracy都不錯，模型收斂佳
5. 寫了調用模型的程式，支援將影片每隔n秒擷取一幀作為辨識輸入源
6. 輸出格式如下:
<img width="406" height="229" alt="image" src="https://github.com/user-attachments/assets/bc588f7a-b8b3-4ac4-b488-924276a13589" />


## 目前問題:

1. 資料量太少太單一
2. 沒有足夠算力支援處理影片，所以只能採用截取幀的方式來做圖象訓練

## 未來可期:

1. 增加更多不同類型的圖片，應付各種不同的情景
2. 如果圖片夠多了，可嘗試更大參數的模型(ex.ResNet50)
3. 定量
