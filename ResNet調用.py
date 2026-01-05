import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ================= 設定區 =================
VIDEO_PATH = r"C:\Users\User\Downloads\Polydisperse droplets (1 ppm PFOA) .mp4"
MODEL_PATH = '0105_test-4.pth'
INTERVAL_SECONDS = 10
# =========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入模型
checkpoint = torch.load(MODEL_PATH, map_location=device)
classes = checkpoint['classes']

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device).eval()

# 預處理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 處理影片
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_interval = int(fps * INTERVAL_SECONDS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"影片 FPS: {fps:.2f}, 每 {INTERVAL_SECONDS} 秒檢測一次")

for frame_pos in range(0, total_frames, frame_interval):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = cap.read()
    if not ret:
        break
    
    # 預處理並預測
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(input_tensor)[0], dim=0)
    
    pred_idx = torch.argmax(probs).item()
    timestamp = frame_pos / fps
    
    print(f"時間: {timestamp:7.2f} 秒 | 預測: {classes[pred_idx]:<10} (信心度: {probs[pred_idx]:.4f})")

cap.release()
print("檢測完成")