##整體model架構

import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class LabelStudioDataset(Dataset):
    def __init__(self, json_file, img_root_dir, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.img_root_dir = img_root_dir
        self.transform = transform

        # 建立標籤映射 (例如: {'Change': 0, 'Nochange': 1})
        # 這裡會自動抓取所有出現過的 choice 建立字典
        self.classes = list(set(item['choice'] for item in self.data))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 處理圖片路徑
        rel_path = item['image'].replace('/data/upload/7/', '')
        img_path = os.path.join(self.img_root_dir, rel_path)

        image = Image.open(img_path).convert('RGB')

        # 處理標籤
        label_name = item['choice']
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def train_model(model, dataloader, val_loader, criterion, optimizer, EPOCHS):
  #用GPU訓練，不然就用CPU
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  history = {
      'train_loss': [],
      'train_acc': [],
      'test_loss': [],
      'test_acc': []
      }

  for epoch in range(EPOCHS):
    #訓練模式
    model.train()

    #初始化
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 歸零梯度
        optimizer.zero_grad()

        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向傳播與更新
        loss.backward()
        optimizer.step()

        # 統計數據
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    #推論測試集
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
      for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)
    history['test_loss'].append(val_loss)
    history['test_acc'].append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}% | Test - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

  return history