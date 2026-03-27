import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch

# confusion matrix
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, colorbar=True)

    ax.set_title('Confusion Matrix', fontsize=24, pad=20)
    ax.set_xlabel("Predicted label", fontsize=20)
    ax.set_ylabel("True label", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)

    for text in disp.text_.ravel():
        text.set_fontsize(24)

    cbar = ax.images[-1].colorbar
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.show()

    return fig

# accuracy 曲線
def plot_accuracy_curves(history):
    epochs = range(1, len(history['train_acc']) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, history['train_acc'], label='Train Accuracy')
    ax.plot(epochs, history['test_acc'], label='Test Accuracy')

    ax.set_title('Training & Testing Accuracy', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(epochs)

    ax.legend(fontsize=11)
    ax.grid(False)

    fig.tight_layout()
    plt.show()

    return fig

# loss 曲線
def plot_loss_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, history['train_loss'], label='Train Loss')
    ax.plot(epochs, history['test_loss'], label='Test Loss')

    ax.set_title('Training & Testing Loss', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xticks(epochs)
    
    ax.legend(fontsize=11)
    ax.grid(False)

    fig.tight_layout()
    plt.show()

    return fig

# roc 曲線
def plot_roc_curve(model, dataloader, device, dataset_classes):
    model.eval()
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    
    ax.set_title('ROC Curve', fontsize=14)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(False)
    
    fig.tight_layout()
    plt.show()
    
    return fig