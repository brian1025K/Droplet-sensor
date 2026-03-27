import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
import torchclass as tc
import evaluate as ev
import plot as pl

# 設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JSON_FILE = './0327_labeling.json'
IMG_ROOT = './images'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4  # 全面微調時，LR 必須小 (1e-4 ~ 1e-5)
TRAIN_RATIO = 0.6


# 數據準備
# 加入 Normalize
# 資料增強
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomRotation(degrees=15),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 測試集不使用資料增強，只做基本轉換
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 先載入完整數據集用於分割
full_dataset = tc.LabelStudioDataset(json_file=JSON_FILE, img_root_dir=IMG_ROOT, transform=transform)

# 計算訓練集和測試集大小
dataset_size = len(full_dataset)
train_size = int(TRAIN_RATIO * dataset_size)
test_size = dataset_size - train_size

# 分割數據集
train_dataset, test_dataset_temp = random_split(full_dataset, [train_size, test_size])

# 為測試集創建使用不同transform的dataset
test_dataset_full = tc.LabelStudioDataset(json_file=JSON_FILE, img_root_dir=IMG_ROOT, transform=test_transform)
# 使用相同的索引創建測試集
test_indices = test_dataset_temp.indices
test_dataset = torch.utils.data.Subset(test_dataset_full, test_indices)

# 創建 DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

dataset = full_dataset
dataloader = train_dataloader

print(f"數據集載入完成，共有 {len(dataset)} 張圖片，類別: {dataset.classes}")

# 模型建置
# 載入預訓練模型、權重
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 修改最後全連接層
num_classes = len(dataset.classes)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model = model.to(DEVICE)

# Loss 與優化器
criterion = nn.CrossEntropyLoss()

# 優化器監聽 model.parameters() (全部層)，而不僅僅是全連接層
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 訓練
model = model.to(DEVICE)
history = tc.train_model(model, dataloader, test_dataloader, criterion, optimizer, EPOCHS)

# 最終評估 (使用測試集)
print("\n最終模型評估 (測試集):")
ev.evaluate_model(model, test_dataloader, DEVICE, num_classes, dataset.classes)

# 繪製訓練曲線
pl.plot_accuracy_curves(history)
pl.plot_loss_curves(history)
pl.plot_roc_curve(model, test_dataloader, DEVICE, dataset.classes)

# 除存模型路徑
model_save_path = '0327_training.pth'

# 儲存模型狀態字典和相關資訊
torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': dataset.class_to_idx,
    'classes': dataset.classes
}, model_save_path)

print(f"模型已成功儲存至: {model_save_path}")