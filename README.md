# 服務於Droplet sensor的圖象辨識

## Workflow:

從影片中擷取圖片 -> 資料標記(labeling) -> 模型訓練/微調 -> 調用model做推理

## 使用ResNet pre-trained model做圖片辨識
1. 使用pytorch作為本次的核心程式庫
2. 微調了ResNet18模型，用於辨識是否有變化
3. 因為圖片與預訓練的圖片差異極大，所以採取了全微調(Full fine-tuning)策略
4. 圖片沒有很多，且不同樣本間差異小，所以有使用數據增強(隨機旋轉，隨機明暗度等)
5. 看Loss及Accuracy都不錯，模型收斂佳
6. 寫了調用模型的程式，支援將影片每隔n秒擷取一幀作為辨識輸入源
7. 輸出格式如下:
<img width="406" height="229" alt="image" src="https://github.com/user-attachments/assets/bc588f7a-b8b3-4ac4-b488-924276a13589" />


## 目前問題:

1. 資料太少太單一，導致模型在推論非訓練集的影片時準確度會變很差
2. 沒有足夠算力支援處理影片，所以只能採用截取幀的方式來做圖象訓練

## 未來可期:

1. 增加更多不同類型的圖片，應付各種不同的情景
2. 如果圖片夠多了，可嘗試更大參數的模型(ex.ResNet50)
3. 定量問題

## 參考資料:

Fine-Tuning a Pre-Trained ResNet-18 Model for Image Classification on Custom Dataset with PyTorch

https://medium.com/@imabhi1216/fine-tuning-a-pre-trained-resnet-18-model-for-image-classification-on-custom-dataset-with-pytorch-02df12e83c2c

Deep Residual Learning for Image Recognition

https://arxiv.org/pdf/1512.03385

直觀理解ResNet —簡介、 觀念及實作(Python Keras)

https://medium.com/@rossleecooloh/%E7%9B%B4%E8%A7%80%E7%90%86%E8%A7%A3resnet-%E7%B0%A1%E4%BB%8B-%E8%A7%80%E5%BF%B5%E5%8F%8A%E5%AF%A6%E4%BD%9C-python-keras-8d1e2e057de2
