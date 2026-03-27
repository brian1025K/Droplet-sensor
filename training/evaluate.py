##模型評估

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import plot as pl

def evaluate_model(model, dataloader, device, num_classes, dataset_classes):
    """
    評估模型性能
    """
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 計算準確率
    accuracy = 100 * sum(np.array(all_labels) == np.array(all_predictions)) / len(all_labels)

    # 計算各項指標
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)

    print("\n" + "="*50)
    print("測試集評估結果")
    print("="*50)
    print(f"準確率 (Accuracy): {accuracy:.2f}%")
    print(f"精確率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分數 (F1-Score): {f1:.4f}")
    print("\n混淆矩陣 (Confusion Matrix):")
    print(f"類別順序: {dataset_classes}")
    print(cm)
    print("="*50)

    pl.plot_confusion_matrix(cm, dataset_classes)

    return accuracy, precision, recall, f1