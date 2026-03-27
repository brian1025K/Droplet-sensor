# 服務於Droplet sensor的圖像辨識

Language option:
- [English](README.en.md)

## Workflow

從影片中擷取圖片 -> 資料標記(labeling) -> 模型訓練/微調 -> 調用model做未知影片的推理

## 簡介
1. 使用ResNet18 pre-trained model做圖片辨識
2. 因為圖片與預訓練的圖片差異極大，所以採取了全微調(Full fine-tuning)策略
3. ~~圖片沒有很多，且不同樣本間差異小，所以有使用數據增強(隨機旋轉，隨機明暗度等)~~(已解決)
4. 寫了調用模型的程式，支援將影片每隔n秒擷取一幀作為辨識輸入源
5. 輸出格式如下:
   
   <img width="406" height="229" alt="image" src="https://github.com/user-attachments/assets/bc588f7a-b8b3-4ac4-b488-924276a13589" />

## 目前結果

1. 圖片量夠多，關閉數據增強

2. 目前模型預測結果趨於穩定，準確度高

<img width="800" height="600" alt="0327cm" src="https://github.com/user-attachments/assets/b3686990-5962-447b-8afb-8a0465c84ef0" />

## 目前問題

1. ~~資料太少太單一，導致模型在推論非訓練集的影片時準確度會變很差(逐漸改善中)~~(已解決)
2. 沒有足夠算力支援處理影片，所以只能採用截取幀的方式來做圖象訓練
3. ~~目前把變化的時間點直接打在影片內，可能會影響模型的判斷(待驗證)~~(已解決)

## 未來可期

1. 增加更多不同類型的圖片，應付各種不同的情景 ✅(已經夠了)
2. 如果圖片夠多了，可嘗試更大參數的模型(ex.ResNet50)

## 更新日誌

2026/2/5 

1. 劃分出訓練集&測試集，並增加confusion matrix等評估用指標。增加了一些圖片做訓練(共162張)

2026/3/27 

1. 拆分/重構程式(Traing資料夾)

2. 增加data量(目前共379張照片)

3. 新增Training時的ACC曲線，Loss曲線及ROC曲線

## 參考資料

Fine-Tuning a Pre-Trained ResNet-18 Model for Image Classification on Custom Dataset with PyTorch

https://medium.com/@imabhi1216/fine-tuning-a-pre-trained-resnet-18-model-for-image-classification-on-custom-dataset-with-pytorch-02df12e83c2c

Deep Residual Learning for Image Recognition

https://arxiv.org/pdf/1512.03385

直觀理解ResNet —簡介、 觀念及實作(Python Keras)

https://medium.com/@rossleecooloh/%E7%9B%B4%E8%A7%80%E7%90%86%E8%A7%A3resnet-%E7%B0%A1%E4%BB%8B-%E8%A7%80%E5%BF%B5%E5%8F%8A%E5%AF%A6%E4%BD%9C-python-keras-8d1e2e057de2

tif 2 jpg

https://github.com/santoshkaranam/convertTif2PngOrJpg
