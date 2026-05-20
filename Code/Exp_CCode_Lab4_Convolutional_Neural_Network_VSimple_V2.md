---
jupyter:
  accelerator: GPU
  colab:
    gpuType: T4
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="Trajy9UYPM35"}
# Instalasi dan Verifikasi Dependensi
:::

::: {.cell .code id="XyzFm-WjdmgJ"}
``` python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Cek versi PyTorch
print(f"PyTorch Version : {torch.__version__}")
print(f"Torchvision     : {torchvision.__version__}")

# Cek ketersediaan GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device          : {device}")
```
:::

::: {.cell .code id="xcwRFHh_dqXm"}
``` python
# ============================
# 3.2.A. KONVOLUSI MANUAL
# ============================

# Input gambar 5×5 (1 channel, 1 batch)
input_image = torch.tensor([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1]
], dtype=torch.float32).reshape(1, 1, 5, 5)  # (batch, channel, H, W)

print("Input Image (5×5):")
print(input_image.squeeze())
print(f"Shape: {input_image.shape}")  # torch.Size([1, 1, 5, 5])

# ============================
# MEMBUAT CONV LAYER
# ============================

# Conv2d: 1 input channel, 1 output channel, kernel 3×3
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,
                        stride=1, padding=0, bias=False)

# Set filter secara manual (untuk edukasi)
custom_kernel = torch.tensor([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
], dtype=torch.float32).reshape(1, 1, 3, 3)

conv_layer.weight = nn.Parameter(custom_kernel)

# Terapkan konvolusi
output = conv_layer(input_image)
print("\nFilter (3×3):")
print(custom_kernel.squeeze())
print(f"\nOutput Feature Map ({output.shape[-2]}×{output.shape[-1]}):")
print(output.squeeze().detach())
# Input 5×5, Kernel 3×3, No Padding → Output: (5-3)/1 + 1 = 3×3
```
:::

::: {.cell .code id="i_uJ3Nbrdsym"}
``` python
# ============================
# 3.2.B. EFEK PADDING DAN STRIDE
# ============================


# ============================
# DEMONSTRASI PADDING
# ============================

# Tanpa padding: ukuran output mengecil
conv_no_pad = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
conv_no_pad.weight = nn.Parameter(custom_kernel.clone())
out_no_pad = conv_no_pad(input_image)

# Dengan padding=1: ukuran output sama dengan input
conv_with_pad = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
conv_with_pad.weight = nn.Parameter(custom_kernel.clone())
out_with_pad = conv_with_pad(input_image)

print("=== EFEK PADDING ===")
print(f"Input shape          : {input_image.shape[-2]}×{input_image.shape[-1]}")
print(f"Tanpa padding (p=0)  : {out_no_pad.shape[-2]}×{out_no_pad.shape[-1]}")
print(f"Dengan padding (p=1) : {out_with_pad.shape[-2]}×{out_with_pad.shape[-1]}")

# ============================
# DEMONSTRASI STRIDE
# ============================

# Stride=1 (default)
conv_s1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
conv_s1.weight = nn.Parameter(custom_kernel.clone())
out_s1 = conv_s1(input_image)

# Stride=2
conv_s2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=0, bias=False)
conv_s2.weight = nn.Parameter(custom_kernel.clone())
out_s2 = conv_s2(input_image)

print("\n=== EFEK STRIDE ===")
print(f"Input shape    : {input_image.shape[-2]}×{input_image.shape[-1]}")
print(f"Stride=1 output: {out_s1.shape[-2]}×{out_s1.shape[-1]}")
print(f"Stride=2 output: {out_s2.shape[-2]}×{out_s2.shape[-1]}")
```
:::

::: {.cell .code id="QPLK6-KId2lh"}
``` python
# ============================
# 3.2.C. FILTER DETEKSI TEPI PADA GAMBAR MNIST
# ============================

# Download 1 gambar MNIST
transform = transforms.Compose([transforms.ToTensor()])
mnist_sample = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Ambil 1 gambar
sample_image, sample_label = mnist_sample[0]
print(f"Gambar digit: {sample_label}")
print(f"Shape: {sample_image.shape}")  # (1, 28, 28)

# Definisikan berbagai filter deteksi tepi
filters = {
    'Tepi Vertikal': torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32),
    'Tepi Horizontal': torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32),
    'Sobel X': torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32),
    'Sobel Y': torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32),
    'Sharpen': torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32),
    'Blur (Box)': torch.ones(3, 3, dtype=torch.float32) / 9.0,
}

# Terapkan setiap filter
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle(f'Efek Berbagai Filter pada Digit "{sample_label}"',
             fontsize=16, fontweight='bold', y=1.02)

# Gambar asli
axes[0, 0].imshow(sample_image.squeeze(), cmap='gray')
axes[0, 0].set_title('Gambar Asli', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Kosongkan sisa posisi pertama
axes[1, 0].axis('off')

# Terapkan filter
input_batch = sample_image.unsqueeze(0)  # (1, 1, 28, 28)

for idx, (name, kernel) in enumerate(filters.items()):
    conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    conv.weight = nn.Parameter(kernel.reshape(1, 1, 3, 3))

    with torch.no_grad():
        output = conv(input_batch)

    row = (idx + 1) // 4
    col = (idx + 1) % 4
    axes[row, col].imshow(output.squeeze().numpy(), cmap='gray')
    axes[row, col].set_title(f'Filter: {name}', fontsize=11, fontweight='bold')
    axes[row, col].axis('off')

# Sembunyikan axes kosong
axes[1, 3].axis('off')

plt.tight_layout(h_pad=1)
plt.show()
```
:::

::: {.cell .code id="Pt6Nugf4d38i"}
``` python
# ============================
# 3.2.D. DEMONSTRASI MAX POOLING
# ============================

# Buat input 4×4
pool_input = torch.tensor([
    [1, 3, 2, 4],
    [5, 6, 7, 8],
    [3, 2, 1, 0],
    [1, 0, 3, 4]
], dtype=torch.float32).reshape(1, 1, 4, 4)

# Max Pooling 2×2
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
pool_output = max_pool(pool_input)

print("Input (4×4):")
print(pool_input.squeeze())
print(f"\nSetelah MaxPool2d(2) → Output (2×2):")
print(pool_output.squeeze())

# Visualisasi pada gambar MNIST
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Gambar asli
axes[0].imshow(sample_image.squeeze(), cmap='gray')
axes[0].set_title(f'Asli (28×28)', fontsize=13, fontweight='bold')
axes[0].axis('off')

# Setelah MaxPool 2×2
pool2 = nn.MaxPool2d(2)
pooled_2 = pool2(sample_image.unsqueeze(0))
axes[1].imshow(pooled_2.squeeze(), cmap='gray')
axes[1].set_title(f'MaxPool2d(2) → 14×14', fontsize=13, fontweight='bold')
axes[1].axis('off')

# Setelah MaxPool 4×4
pool4 = nn.MaxPool2d(4)
pooled_4 = pool4(sample_image.unsqueeze(0))
axes[2].imshow(pooled_4.squeeze(), cmap='gray')
axes[2].set_title(f'MaxPool2d(4) → 7×7', fontsize=13, fontweight='bold')
axes[2].axis('off')

plt.suptitle('Efek Max Pooling pada Gambar',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```
:::

::: {.cell .code id="Cxomb5Ckd5dZ"}
``` python
# ============================
# 3.3.A. DATASET MNIST
# ============================

torch.manual_seed(42)

# Transform: konversi ke tensor & normalisasi
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download dataset
train_dataset = datasets.MNIST(root='./data', train=True,
                                download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False,
                               download=True, transform=transform)

# DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples : {len(train_dataset)}")
print(f"Test samples     : {len(test_dataset)}")
print(f"Image shape      : {train_dataset[0][0].shape}")  # (1, 28, 28)
print(f"Batch size       : {batch_size}")
print(f"Training batches : {len(train_loader)}")
```
:::

::: {.cell .code id="mJfJ81eGd61b"}
``` python
# ============================
# 3.3.B. VISUALISASI SAMPEL MNIST
# ============================

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('Sampel Dataset MNIST', fontsize=16, fontweight='bold', y=1.02)

for i, ax in enumerate(axes.flat):
    image, label = train_dataset[i]
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}', fontsize=12, fontweight='bold')
    ax.axis('off')

plt.tight_layout(h_pad=2.5)
plt.show()
```
:::

::: {.cell .code id="Y-87o2iqd8X4"}
``` python
# ============================
# 3.3.C. MODEL CNN UNTUK MNIST
# ============================

# Arsitektur:
#   Input (1, 28, 28)
#     ↓
#   Conv2d(1→16, 3×3, pad=1) + ReLU + MaxPool(2)  → (16, 14, 14)
#     ↓
#   Conv2d(16→32, 3×3, pad=1) + ReLU + MaxPool(2) → (32, 7, 7)
#     ↓
#   Flatten → (32 × 7 × 7 = 1568)
#     ↓
#   Linear(1568→128) + ReLU + Dropout(0.5)
#     ↓
#   Linear(128→10)

model_cnn_mnist = nn.Sequential(
    # === Blok Konvolusi 1 ===
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),  # (1,28,28)→(16,28,28)
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # (16,28,28)→(16,14,14)

    # === Blok Konvolusi 2 ===
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # (16,14,14)→(32,14,14)
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # (32,14,14)→(32,7,7)

    # === Classifier ===
    nn.Flatten(),                  # (32,7,7) → (1568)
    nn.Linear(32 * 7 * 7, 128),   # 1568 → 128
    nn.ReLU(),
    nn.Dropout(0.5),               # Dropout 50%
    nn.Linear(128, 10)             # 128 → 10 kelas (TANPA Softmax!)
)

# Pindahkan ke device
model_cnn_mnist = model_cnn_mnist.to(device)

print("Arsitektur Model CNN MNIST:")
print(model_cnn_mnist)

total_params = sum(p.numel() for p in model_cnn_mnist.parameters())
print(f"\nTotal parameter: {total_params:,}")
```
:::

::: {.cell .code id="1MvMt9Gdd-Bw"}
``` python
# ============================
# 3.3.E. VERIFIKASI DIMENSI OUTPUT
# ============================

# Buat dummy input
dummy_input = torch.randn(1, 1, 28, 28).to(device)

# Forward pass untuk melihat output
with torch.no_grad():
    dummy_output = model_cnn_mnist(dummy_input)

print(f"Input shape : {dummy_input.shape}")   # (1, 1, 28, 28)
print(f"Output shape: {dummy_output.shape}")   # (1, 10)

# Verifikasi layer per layer
print("\n--- Dimensi per Layer ---")

# Simulasi manual per layer (di CPU untuk kemudahan inspeksi)
x = torch.randn(1, 1, 28, 28)
print(f"Input            : {x.shape}")      # (1, 1, 28, 28)

# Blok Konvolusi 1
x = nn.Conv2d(1, 16, 3, padding=1)(x);  print(f"Conv2d(1→16)     : {x.shape}")  # (1,16,28,28)
x = nn.ReLU()(x);                        print(f"ReLU             : {x.shape}")  # (1,16,28,28)
x = nn.MaxPool2d(2)(x);                  print(f"MaxPool2d(2)     : {x.shape}")  # (1,16,14,14)

# Blok Konvolusi 2
x = nn.Conv2d(16, 32, 3, padding=1)(x);  print(f"Conv2d(16→32)    : {x.shape}")  # (1,32,14,14)
x = nn.ReLU()(x);                         print(f"ReLU             : {x.shape}")  # (1,32,14,14)
x = nn.MaxPool2d(2)(x);                   print(f"MaxPool2d(2)     : {x.shape}")  # (1,32,7,7)

# Classifier
x = nn.Flatten()(x);                      print(f"Flatten          : {x.shape}")  # (1,1568)
x = nn.Linear(1568, 128)(x);              print(f"Linear(1568→128) : {x.shape}")  # (1,128)
x = nn.Linear(128, 10)(x);                print(f"Linear(128→10)   : {x.shape}")  # (1,10)
```
:::

::: {.cell .code id="biWbgkNGeApK"}
``` python
# ============================
# LOSS FUNCTION & OPTIMIZER
# ============================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn_mnist.parameters(), lr=0.001)

# ============================
# 3.3.F. TRAINING LOOP
# ============================

num_epochs = 10
train_losses = []
test_losses = []
train_accs = []
test_accs = []

print("Mulai Training CNN MNIST...")
print("=" * 75)

for epoch in range(num_epochs):
    # ---- TRAINING PHASE ----
    model_cnn_mnist.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model_cnn_mnist(images)
        loss = criterion(outputs, labels)

        # Backward pass & update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total * 100
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ---- TESTING PHASE ----
    model_cnn_mnist.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model_cnn_mnist(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = correct / total * 100
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

print("=" * 75)
print(f"Akurasi Akhir → Train: {train_accs[-1]:.2f}% | Test: {test_accs[-1]:.2f}%")
```
:::

::: {.cell .code id="LtOLTuNPeDBK"}
``` python
# ============================
# 3.3.G. PLOT LOSS & ACCURACY
# ============================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Loss
axes[0].plot(range(1, num_epochs+1), train_losses, 'o-',
             color='crimson', linewidth=2, label='Train Loss')
axes[0].plot(range(1, num_epochs+1), test_losses, 's--',
             color='steelblue', linewidth=2, label='Test Loss')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training vs Test Loss — CNN MNIST', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(range(1, num_epochs+1), train_accs, 'o-',
             color='crimson', linewidth=2, label='Train Accuracy')
axes[1].plot(range(1, num_epochs+1), test_accs, 's--',
             color='steelblue', linewidth=2, label='Test Accuracy')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Train vs Test Accuracy — CNN MNIST', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```
:::

::: {.cell .code id="cb2-aDlXeDwy"}
``` python
# ============================
# 3.3.H. EVALUASI DETAIL
# ============================

model_cnn_mnist.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model_cnn_mnist(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Classification Report
print("\nClassification Report — CNN MNIST:")
print("=" * 60)
print(classification_report(all_labels, all_preds,
                             target_names=[f'Digit {i}' for i in range(10)]))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)])
plt.xlabel('Prediksi', fontsize=13)
plt.ylabel('Aktual', fontsize=13)
plt.title('Confusion Matrix — CNN MNIST', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
```
:::

::: {.cell .code id="qS9dytToeGJh"}
``` python
# ============================
# 3.3.I. VISUALISASI PREDIKSI
# ============================

model_cnn_mnist.eval()
test_images, test_labels = next(iter(test_loader))
test_images_device = test_images.to(device)

with torch.no_grad():
    outputs = model_cnn_mnist(test_images_device)
    probabilities = torch.softmax(outputs, dim=1)
    _, predictions = torch.max(outputs, 1)

# Plot 10 sampel
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle('Prediksi CNN pada Data Test MNIST',
             fontsize=16, fontweight='bold', y=1.02)

for i, ax in enumerate(axes.flat):
    img = test_images[i].squeeze().cpu().numpy()
    true_label = test_labels[i].item()
    pred_label = predictions[i].cpu().item()
    confidence = probabilities[i][pred_label].cpu().item() * 100

    ax.imshow(img, cmap='gray')

    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f'Pred: {pred_label} ({confidence:.1f}%)\nTrue: {true_label}',
                 fontsize=11, fontweight='bold', color=color)
    ax.axis('off')

plt.tight_layout(h_pad=2.7)
plt.show()
```
:::

::: {.cell .code id="rcagIfvveH3R"}
``` python
# ============================
# 3.4.B. DATASET CIFAR-10
# ============================

torch.manual_seed(42)

# Transform untuk training (dengan augmentasi sederhana)
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),   # Mean per channel (R,G,B)
                          (0.2470, 0.2435, 0.2616)),  # Std per channel (R,G,B)
])

# Transform untuk testing (tanpa augmentasi)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                          (0.2470, 0.2435, 0.2616)),
])

# Download dataset
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                   download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                  download=True, transform=transform_test)

# DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Nama kelas
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print(f"Training samples : {len(train_dataset)}")
print(f"Test samples     : {len(test_dataset)}")
print(f"Image shape      : {train_dataset[0][0].shape}")  # (3, 32, 32)
print(f"Classes          : {class_names}")
```
:::

::: {.cell .code id="Ko74fPkHeJ2i"}
``` python
# ============================
# 3.4.C. VISUALISASI SAMPEL CIFAR-10
# ============================

# Untuk de-normalisasi gambar agar bisa ditampilkan
# (kita balikkan normalisasi: img = img * std + mean)
mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
std = torch.tensor([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle('Sampel Dataset CIFAR-10', fontsize=16, fontweight='bold', y=1.02)

for i, ax in enumerate(axes.flat):
    image, label = train_dataset[i]
    # De-normalisasi
    img_display = image * std + mean
    img_display = img_display.clamp(0, 1)
    # Ubah dari (C, H, W) ke (H, W, C) untuk matplotlib
    img_display = img_display.permute(1, 2, 0).numpy()

    ax.imshow(img_display)
    ax.set_title(f'{class_names[label]}', fontsize=12, fontweight='bold')
    ax.axis('off')

plt.tight_layout(h_pad=2.5)
plt.show()
```
:::

::: {.cell .code id="iiijhjWEeMLA"}
``` python
# ============================
# 3.4.D. MODEL CNN UNTUK CIFAR-10
# ============================

# Arsitektur yang lebih dalam untuk gambar berwarna:
#
#   Input (3, 32, 32) — gambar RGB
#     ↓
#   Conv2d(3→32, 3×3, pad=1) + ReLU → (32, 32, 32)
#   Conv2d(32→32, 3×3, pad=1) + ReLU → (32, 32, 32)
#   MaxPool(2) → (32, 16, 16)
#   Dropout2d(0.25)
#     ↓
#   Conv2d(32→64, 3×3, pad=1) + ReLU → (64, 16, 16)
#   Conv2d(64→64, 3×3, pad=1) + ReLU → (64, 16, 16)
#   MaxPool(2) → (64, 8, 8)
#   Dropout2d(0.25)
#     ↓
#   Conv2d(64→128, 3×3, pad=1) + ReLU → (128, 8, 8)
#   Conv2d(128→128, 3×3, pad=1) + ReLU → (128, 8, 8)
#   MaxPool(2) → (128, 4, 4)
#   Dropout2d(0.25)
#     ↓
#   Flatten → (128 × 4 × 4 = 2048)
#     ↓
#   Linear(2048→512) + ReLU + Dropout(0.5)
#   Linear(512→10)

model_cnn_cifar = nn.Sequential(
    # === Blok 1: 3 → 32 channel ===
    nn.Conv2d(3, 32, kernel_size=3, padding=1),    # (3,32,32) → (32,32,32)
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),   # (32,32,32) → (32,32,32)
    nn.ReLU(),
    nn.MaxPool2d(2),                                # (32,32,32) → (32,16,16)
    nn.Dropout2d(0.25),

    # === Blok 2: 32 → 64 channel ===
    nn.Conv2d(32, 64, kernel_size=3, padding=1),   # (32,16,16) → (64,16,16)
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),   # (64,16,16) → (64,16,16)
    nn.ReLU(),
    nn.MaxPool2d(2),                                # (64,16,16) → (64,8,8)
    nn.Dropout2d(0.25),

    # === Blok 3: 64 → 128 channel ===
    nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (64,8,8) → (128,8,8)
    nn.ReLU(),
    nn.Conv2d(128, 128, kernel_size=3, padding=1), # (128,8,8) → (128,8,8)
    nn.ReLU(),
    nn.MaxPool2d(2),                                # (128,8,8) → (128,4,4)
    nn.Dropout2d(0.25),

    # === Classifier ===
    nn.Flatten(),                    # (128,4,4) → (2048)
    nn.Linear(128 * 4 * 4, 512),    # 2048 → 512
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)              # 512 → 10 kelas
)

model_cnn_cifar = model_cnn_cifar.to(device)

print("Arsitektur Model CNN CIFAR-10:")
print(model_cnn_cifar)

total_params = sum(p.numel() for p in model_cnn_cifar.parameters())
print(f"\nTotal parameter: {total_params:,}")
```
:::

::: {.cell .code id="i3lBhmf0eNhZ"}
``` python
# ============================
# 3.4.E. VERIFIKASI DIMENSI
# ============================

dummy = torch.randn(1, 3, 32, 32).to(device)
with torch.no_grad():
    out = model_cnn_cifar(dummy)

print(f"Input shape : {dummy.shape}")
print(f"Output shape: {out.shape}")

# Simulasi per layer
print("\n--- Dimensi per Layer ---")
x = torch.randn(1, 3, 32, 32)
print(f"Input              : {x.shape}")

# Blok 1
x = nn.Conv2d(3, 32, 3, padding=1)(x);     print(f"Conv2d(3→32)       : {x.shape}")
x = nn.ReLU()(x);                           print(f"ReLU               : {x.shape}")
x = nn.Conv2d(32, 32, 3, padding=1)(x);    print(f"Conv2d(32→32)      : {x.shape}")
x = nn.ReLU()(x);                           print(f"ReLU               : {x.shape}")
x = nn.MaxPool2d(2)(x);                     print(f"MaxPool2d(2)       : {x.shape}")

# Blok 2
x = nn.Conv2d(32, 64, 3, padding=1)(x);    print(f"Conv2d(32→64)      : {x.shape}")
x = nn.ReLU()(x);                           print(f"ReLU               : {x.shape}")
x = nn.Conv2d(64, 64, 3, padding=1)(x);    print(f"Conv2d(64→64)      : {x.shape}")
x = nn.ReLU()(x);                           print(f"ReLU               : {x.shape}")
x = nn.MaxPool2d(2)(x);                     print(f"MaxPool2d(2)       : {x.shape}")

# Blok 3
x = nn.Conv2d(64, 128, 3, padding=1)(x);   print(f"Conv2d(64→128)     : {x.shape}")
x = nn.ReLU()(x);                           print(f"ReLU               : {x.shape}")
x = nn.Conv2d(128, 128, 3, padding=1)(x);  print(f"Conv2d(128→128)    : {x.shape}")
x = nn.ReLU()(x);                           print(f"ReLU               : {x.shape}")
x = nn.MaxPool2d(2)(x);                     print(f"MaxPool2d(2)       : {x.shape}")

# Classifier
x = nn.Flatten()(x);                        print(f"Flatten            : {x.shape}")
x = nn.Linear(2048, 512)(x);                print(f"Linear(2048→512)   : {x.shape}")
x = nn.Linear(512, 10)(x);                  print(f"Linear(512→10)     : {x.shape}")
```
:::

::: {.cell .code id="BJfRHuW9eOvR"}
``` python
# ============================
# 3.4.F. LOSS & OPTIMIZER
# ============================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn_cifar.parameters(), lr=0.001)

# ============================
# TRAINING LOOP
# ============================

num_epochs = 15
train_losses = []
test_losses = []
train_accs = []
test_accs = []

print("Mulai Training CNN CIFAR-10...")
print("=" * 80)

for epoch in range(num_epochs):
    # ---- TRAINING PHASE ----
    model_cnn_cifar.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model_cnn_cifar(images)
        loss = criterion(outputs, labels)

        # Backward pass & update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total * 100
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ---- TESTING PHASE ----
    model_cnn_cifar.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model_cnn_cifar(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = correct / total * 100
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    # Print setiap 5 epoch
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

print("=" * 80)
print(f"Akurasi Akhir → Train: {train_accs[-1]:.2f}% | Test: {test_accs[-1]:.2f}%")
```
:::

::: {.cell .code id="ncgUfNMNeRgR"}
``` python
# ============================
# 3.4.G. PLOT LOSS & ACCURACY
# ============================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Loss
axes[0].plot(range(1, num_epochs+1), train_losses, '-',
             color='crimson', linewidth=2, label='Train Loss')
axes[0].plot(range(1, num_epochs+1), test_losses, '--',
             color='steelblue', linewidth=2, label='Test Loss')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training vs Test Loss — CNN CIFAR-10', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(range(1, num_epochs+1), train_accs, '-',
             color='crimson', linewidth=2, label='Train Accuracy')
axes[1].plot(range(1, num_epochs+1), test_accs, '--',
             color='steelblue', linewidth=2, label='Test Accuracy')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Train vs Test Accuracy — CNN CIFAR-10', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```
:::

::: {.cell .code id="o73paqSMeTEK"}
``` python
# ============================
# 3.4.H. EVALUASI PER KELAS
# ============================

model_cnn_cifar.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model_cnn_cifar(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Classification Report
print("\nClassification Report — CNN CIFAR-10:")
print("=" * 60)
print(classification_report(all_labels, all_preds,
                             target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Prediksi', fontsize=13)
plt.ylabel('Aktual', fontsize=13)
plt.title('Confusion Matrix — CNN CIFAR-10', fontsize=15, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```
:::

::: {.cell .code id="Hvc0nrJfeUh6"}
``` python
# ============================
# 3.4.I. VISUALISASI PREDIKSI
# ============================

model_cnn_cifar.eval()
test_images, test_labels = next(iter(test_loader))
test_images_device = test_images.to(device)

with torch.no_grad():
    outputs = model_cnn_cifar(test_images_device)
    probabilities = torch.softmax(outputs, dim=1)
    _, predictions = torch.max(outputs, 1)

# De-normalisasi untuk visualisasi
mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
std = torch.tensor([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)

fig, axes = plt.subplots(3, 5, figsize=(18, 11))
fig.suptitle('Prediksi CNN pada Data Test CIFAR-10',
             fontsize=16, fontweight='bold', y=1.02)

for i, ax in enumerate(axes.flat):
    img = test_images[i] * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()

    true_label = test_labels[i].item()
    pred_label = predictions[i].cpu().item()
    confidence = probabilities[i][pred_label].cpu().item() * 100

    ax.imshow(img)
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f'Pred: {class_names[pred_label]}\n({confidence:.1f}%)\n'
                 f'True: {class_names[true_label]}',
                 fontsize=10, fontweight='bold', color=color)
    ax.axis('off')

plt.tight_layout(h_pad=1)
plt.show()
```
:::

::: {.cell .code id="HAl-UCKleVLB"}
``` python
# ============================
# 3.4.J. AKURASI PER KELAS
# ============================

# Hitung akurasi per kelas
class_correct = np.zeros(10)
class_total = np.zeros(10)

for label, pred in zip(all_labels, all_preds):
    class_total[label] += 1
    if label == pred:
        class_correct[label] += 1

class_accuracies = class_correct / class_total * 100

# Visualisasi
plt.figure(figsize=(12, 6))
colors = plt.cm.RdYlGn(class_accuracies / 100)  # Merah → Kuning → Hijau
bars = plt.bar(class_names, class_accuracies, color=colors, edgecolor='black', linewidth=0.5)

# Tambahkan label di atas bar
for bar, acc in zip(bars, class_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', va='bottom',
             fontsize=11, fontweight='bold')

plt.xlabel('Kelas', fontsize=12)
plt.ylabel('Akurasi (%)', fontsize=12)
plt.title('Akurasi per Kelas — CNN CIFAR-10', fontsize=14, fontweight='bold')
plt.ylim(0, 105)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```
:::

::: {.cell .code id="d7YymMVDeXAi"}
``` python
# ============================
# 3.5.A. DATASET DENGAN DATA AUGMENTATION
# ============================

torch.manual_seed(42)

# Transform training DENGAN augmentasi
transform_train_aug = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          # Random crop dengan padding
    transforms.RandomHorizontalFlip(p=0.5),        # Flip horizontal 50%
    transforms.ColorJitter(brightness=0.2,         # Variasi warna
                            contrast=0.2,
                            saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                          (0.2470, 0.2435, 0.2616)),
])

# Transform test (TANPA augmentasi)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                          (0.2470, 0.2435, 0.2616)),
])

# Dataset
train_dataset_aug = datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform_train_aug)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                  download=True, transform=transform_test)

# DataLoader
batch_size = 128
train_loader_aug = DataLoader(train_dataset_aug, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples  : {len(train_dataset_aug)}")
print(f"Test samples      : {len(test_dataset)}")
print(f"Batch size        : {batch_size}")
```
:::

::: {.cell .code id="vLKLPhNFeZ1w"}
``` python
# ============================
# 3.5.B. VISUALISASI AUGMENTASI
# ============================

# Ambil 1 gambar asli (tanpa transform)
raw_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                 transform=transforms.ToTensor())
original_image, original_label = raw_dataset[0]

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle(f'Data Augmentation — "{class_names[original_label]}"',
             fontsize=16, fontweight='bold', y=1.02)

# Gambar asli
axes[0, 0].imshow(original_image.permute(1, 2, 0).numpy())
axes[0, 0].set_title('Asli', fontsize=12, fontweight='bold', color='blue')
axes[0, 0].axis('off')

# 9 variasi augmentasi
augment_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

# Konversi ke PIL untuk transforms
from PIL import Image
pil_image = transforms.ToPILImage()(original_image)

for i in range(1, 10):
    row = i // 5
    col = i % 5
    augmented = augment_transform(pil_image)
    axes[row, col].imshow(augmented.permute(1, 2, 0).numpy())
    axes[row, col].set_title(f'Augmentasi {i}', fontsize=11)
    axes[row, col].axis('off')

plt.tight_layout(h_pad=2.5)
plt.show()
```
:::

::: {.cell .code id="RF7wNpXLecAI"}
``` python
# ============================
# 3.5.C. MODEL CNN LANJUTAN (BatchNorm + Dropout)
# ============================

# Pola: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → Pool → Dropout

model_cnn_advanced = nn.Sequential(
    # === Blok 1: 3 → 32 channel ===
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),                              # ← Batch Normalization!
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),

    # === Blok 2: 32 → 64 channel ===
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),

    # === Blok 3: 64 → 128 channel ===
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Conv2d(128, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),

    # === Classifier ===
    nn.Flatten(),
    nn.Linear(128 * 4 * 4, 512),
    nn.BatchNorm1d(512),                             # ← BatchNorm1d untuk FC layer!
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)

model_cnn_advanced = model_cnn_advanced.to(device)

print("Arsitektur Model CNN Lanjutan:")
print(model_cnn_advanced)

total_params = sum(p.numel() for p in model_cnn_advanced.parameters())
print(f"\nTotal parameter: {total_params:,}")
```
:::

::: {.cell .code id="UDjVR6dPedmw"}
``` python
# ============================
# 3.5.D. LOSS, OPTIMIZER, SCHEDULER
# ============================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn_advanced.parameters(), lr=0.001)

# Learning Rate Scheduler: turunkan LR 10× setiap 10 epoch
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ============================
# TRAINING LOOP
# ============================

num_epochs = 15
train_losses = []
test_losses = []
train_accs = []
test_accs = []
learning_rates = []

print("Mulai Training CNN Lanjutan (BatchNorm + Augmentation + Scheduler)...")
print("=" * 85)

for epoch in range(num_epochs):
    # ---- TRAINING PHASE ----
    model_cnn_advanced.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader_aug:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model_cnn_advanced(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader_aug)
    train_acc = correct / total * 100
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Catat learning rate saat ini
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)

    # Step scheduler (setelah setiap epoch)
    scheduler.step()

    # ---- TESTING PHASE ----
    model_cnn_advanced.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model_cnn_advanced(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = correct / total * 100
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

print("=" * 85)
print(f"Akurasi Akhir → Train: {train_accs[-1]:.2f}% | Test: {test_accs[-1]:.2f}%")
```
:::

::: {.cell .code id="41zey_J4egaS"}
``` python
# ============================
# 3.5.E. PLOT LOSS, ACCURACY, DAN LEARNING RATE
# ============================

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Loss
axes[0].plot(range(1, num_epochs+1), train_losses, '-',
             color='crimson', linewidth=2, label='Train Loss')
axes[0].plot(range(1, num_epochs+1), test_losses, '--',
             color='steelblue', linewidth=2, label='Test Loss')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training vs Test Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(range(1, num_epochs+1), train_accs, '-',
             color='crimson', linewidth=2, label='Train Accuracy')
axes[1].plot(range(1, num_epochs+1), test_accs, '--',
             color='steelblue', linewidth=2, label='Test Accuracy')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Train vs Test Accuracy', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

# Learning Rate
axes[2].plot(range(1, num_epochs+1), learning_rates, 'o-',
             color='purple', linewidth=2, markersize=4)
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('Learning Rate', fontsize=12)
axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3)

plt.suptitle('CNN Lanjutan — CIFAR-10 (BatchNorm + Augmentation + Scheduler)',
             fontsize=15, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()
```
:::

::: {.cell .code id="36sheVyleiRh"}
``` python
# ============================
# 3.5.F. EVALUASI & PERBANDINGAN
# ============================

model_cnn_advanced.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model_cnn_advanced(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Classification Report
print("\nClassification Report — CNN Lanjutan CIFAR-10:")
print("=" * 60)
print(classification_report(all_labels, all_preds,
                             target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Prediksi', fontsize=13)
plt.ylabel('Aktual', fontsize=13)
plt.title('Confusion Matrix — CNN Lanjutan CIFAR-10',
          fontsize=15, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```
:::

::: {.cell .code id="AM_ID4hKePYh"}
``` python
# ============================
# 3.5.G. SIMPAN MODEL
# ============================
torch.save(model_cnn_advanced.state_dict(), 'model_cnn_cifar10_advanced.pth')
print("Model CNN lanjutan berhasil disimpan ke 'model_cnn_cifar10_advanced.pth'")

# ============================
# MEMUAT MODEL
# ============================
# Buat arsitektur yang sama
model_loaded = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.MaxPool2d(2), nn.Dropout2d(0.25),

    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(2), nn.Dropout2d(0.25),

    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
    nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
    nn.MaxPool2d(2), nn.Dropout2d(0.25),

    nn.Flatten(),
    nn.Linear(128 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)

model_loaded.load_state_dict(torch.load('model_cnn_cifar10_advanced.pth'))
model_loaded.eval()
print("Model berhasil dimuat!")
```
:::

::: {.cell .code id="DkYpNmy5eqvS"}
``` python
# ============================
# 3.6.A. VISUALISASI FILTER LAYER PERTAMA
# ============================

# Ambil bobot filter dari Conv2d pertama (layer index 0)
# model_cnn_advanced[0] adalah nn.Conv2d(3, 32, 3, padding=1)
first_conv_weights = model_cnn_advanced[0].weight.data.cpu()

print(f"Shape filter layer pertama: {first_conv_weights.shape}")
# (32, 3, 3, 3) → 32 filter, masing-masing 3 channel, 3×3

# Visualisasi 32 filter
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Filter/Kernel Layer Pertama (32 filter, 3×3)',
             fontsize=16, fontweight='bold', y=1.02)

for i, ax in enumerate(axes.flat):
    if i < 32:
        # Ambil filter ke-i, normalisasi untuk visualisasi
        kernel = first_conv_weights[i]
        # Konversi 3-channel kernel ke RGB image
        kernel = kernel.permute(1, 2, 0)  # (3,3,3) → (3,3,3) tapi C last
        # Normalisasi ke [0, 1]
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
        ax.imshow(kernel.numpy())
        ax.set_title(f'Filter {i}', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()
```
:::

::: {.cell .code id="LyAMUmUFes5x"}
``` python
# ============================
# 3.6.B. VISUALISASI FEATURE MAP
# ============================

# Ambil 1 gambar test
sample_img, sample_label = test_dataset[0]
sample_input = sample_img.unsqueeze(0).to(device)  # (1, 3, 32, 32)

# Kita akan mengekstrak output dari setiap blok konvolusi
# Untuk nn.Sequential, kita bisa memproses layer per layer

model_cnn_advanced.eval()

# Dapatkan output dari beberapa layer kunci
feature_maps = {}
x = sample_input

with torch.no_grad():
    for idx, layer in enumerate(model_cnn_advanced):
        x = layer(x)
        # Simpan output setelah ReLU (layer 2, 5 = setelah Conv+BN+ReLU blok 1)
        if idx == 2:   # Setelah ReLU pertama di Blok 1
            feature_maps['Blok 1 - Conv1'] = x.cpu()
        elif idx == 5: # Setelah ReLU kedua di Blok 1
            feature_maps['Blok 1 - Conv2'] = x.cpu()
        elif idx == 10: # Setelah ReLU pertama di Blok 2
            feature_maps['Blok 2 - Conv1'] = x.cpu()
        elif idx == 18: # Setelah ReLU pertama di Blok 3
            feature_maps['Blok 3 - Conv1'] = x.cpu()

# Visualisasi gambar asli
mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
std = torch.tensor([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)

img_display = sample_img * std + mean
img_display = img_display.clamp(0, 1).permute(1, 2, 0).numpy()

# Plot feature maps
for layer_name, fmap in feature_maps.items():
    n_features = min(16, fmap.shape[1])  # Tampilkan max 16 channel

    fig, axes = plt.subplots(2, 9, figsize=(20, 5))
    fig.suptitle(f'Feature Maps: {layer_name} ({fmap.shape[1]} channel, {fmap.shape[2]}×{fmap.shape[3]})',
                 fontsize=14, fontweight='bold', y=1.02)

    # Gambar asli di posisi pertama
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title(f'Asli\n({class_names[sample_label]})',
                         fontsize=9, fontweight='bold', color='blue')
    axes[0, 0].axis('off')

    # Feature maps
    for i in range(1, min(18, n_features + 1)):
        row = i // 9
        col = i % 9
        if i - 1 < n_features:
            axes[row, col].imshow(fmap[0, i-1].numpy(), cmap='viridis')
            axes[row, col].set_title(f'Ch {i-1}', fontsize=8)
        axes[row, col].axis('off')

    # Sembunyikan axes kosong
    for j in range(n_features + 1, 18):
        row = j // 9
        col = j % 9
        axes[row, col].axis('off')

    plt.tight_layout(h_pad=2.5)
    plt.show()
```
:::

::: {.cell .code id="LhkHHxVyex1Z"}
``` python
# ============================
# 3.6.C. PERBANDINGAN FEATURE MAP DI BERBAGAI KEDALAMAN
# ============================

# Ambil 1 feature map paling aktif dari setiap blok
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('Evolusi Feature Map dari Layer Awal ke Layer Dalam',
             fontsize=15, fontweight='bold', y=1.05)

# Gambar asli
axes[0].imshow(img_display)
axes[0].set_title(f'Input\n(32×32)', fontsize=11, fontweight='bold')
axes[0].axis('off')

# Untuk setiap blok, ambil channel dengan aktivasi tertinggi
for i, (name, fmap) in enumerate(feature_maps.items()):
    # Cari channel dengan aktivasi total tertinggi
    channel_activations = fmap[0].sum(dim=(1, 2))  # Sum per channel
    best_channel = channel_activations.argmax().item()

    axes[i+1].imshow(fmap[0, best_channel].numpy(), cmap='hot')
    axes[i+1].set_title(f'{name}\n({fmap.shape[2]}×{fmap.shape[3]}, ch={best_channel})',
                        fontsize=10, fontweight='bold')
    axes[i+1].axis('off')

plt.tight_layout()
plt.show()
```
:::
