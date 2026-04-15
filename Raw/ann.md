# Praktikum: Artificial Neural Network (ANN) Dasar dengan PyTorch

> **Mata Kuliah:** Deep Learning  
> **Topik:** Artificial Neural Network (Feedforward Neural Network)  
> **Framework:** PyTorch  
> **Prasyarat:** Dasar Python, Aljabar Linear dasar, Kalkulus dasar, Machine Learning

---

## Daftar Isi

1. [Pendahuluan](#1-pendahuluan)
2. [Tujuan Praktikum](#2-tujuan-praktikum)
3. [Landasan Teori](#3-landasan-teori)
4. [Persiapan Lingkungan](#4-persiapan-lingkungan)
5. [Praktikum 1 — Tensor & Operasi Dasar PyTorch](#5-praktikum-1--tensor--operasi-dasar-pytorch)
6. [Praktikum 2 — Regresi Linear Sederhana](#6-praktikum-2--regresi-linear-sederhana)
7. [Praktikum 3 — Regresi Non-Linear (Multi-Layer)](#7-praktikum-3--regresi-non-linear-multi-layer)
8. [Praktikum 4 — Klasifikasi Biner](#8-praktikum-4--klasifikasi-biner)
9. [Praktikum 5 — Klasifikasi Multi-Kelas (MNIST)](#9-praktikum-5--klasifikasi-multi-kelas-mnist)
10. [Ringkasan & Best Practices](#10-ringkasan--best-practices)
11. [Latihan Mandiri](#11-latihan-mandiri)
12. [Referensi](#12-referensi)

---

## 1. Pendahuluan

**Artificial Neural Network (ANN)** atau yang sering disebut *Feedforward Neural Network (FNN)* adalah arsitektur jaringan saraf tiruan paling dasar dalam deep learning. ANN menjadi fondasi untuk memahami arsitektur yang lebih kompleks seperti CNN (Convolutional Neural Network), RNN (Recurrent Neural Network), Transformer, dan lainnya.

Pada praktikum ini, kita akan mempelajari bagaimana membangun, melatih, dan mengevaluasi ANN menggunakan **PyTorch** — salah satu framework deep learning paling populer yang dikembangkan oleh Meta AI Research.

> **PENTING:**  
> Seluruh model pada praktikum ini dibangun menggunakan `nn.Sequential()` agar kode lebih ringkas dan mudah dipahami. Pendekatan ini sangat cocok untuk pemula yang baru belajar deep learning.

---

## 2. Tujuan Praktikum

Setelah menyelesaikan praktikum ini, mahasiswa diharapkan mampu:

1. Memahami konsep dasar **neuron buatan**, **layer**, dan **arsitektur ANN**
2. Memahami dan menggunakan **tensor** di PyTorch
3. Membangun model ANN menggunakan `nn.Sequential()`
4. Memahami konsep **forward propagation** dan **backpropagation**
5. Memilih **fungsi aktivasi** yang tepat untuk setiap kasus
6. Memilih **loss function** yang sesuai dengan jenis masalah
7. Melatih model menggunakan **optimizer** (SGD, Adam)
8. Mengevaluasi performa model pada data uji
9. Memvisualisasikan proses pelatihan dan hasil prediksi

---

## 3. Landasan Teori

### 3.1 Neuron Buatan (Artificial Neuron)

Neuron buatan terinspirasi dari neuron biologis di otak manusia. Setiap neuron menerima input, memprosesnya, dan menghasilkan output.

```
    x₁ ──→ w₁ ──╲
    x₂ ──→ w₂ ───→ Σ(wᵢxᵢ + b) ──→ f(z) ──→ output (ŷ)
    x₃ ──→ w₃ ──╱
```

**Secara matematis:**

$$z = \sum_{i=1}^{n} w_i \cdot x_i + b = \mathbf{w}^T \mathbf{x} + b$$

$$\hat{y} = f(z)$$

Di mana:

| Simbol | Keterangan |
|--------|------------|
| $x_i$ | Input ke-i |
| $w_i$ | Bobot (weight) untuk input ke-i |
| $b$ | Bias |
| $z$ | Penjumlahan terboboti (*weighted sum*) |
| $f(z)$ | Fungsi aktivasi |
| $\hat{y}$ | Output / prediksi |

### 3.2 Arsitektur ANN (Feedforward Neural Network)

ANN tersusun dari beberapa **layer** yang saling terhubung:

```
  INPUT LAYER        HIDDEN LAYER(S)         OUTPUT LAYER
  ┌─────────┐       ┌──────────────┐        ┌────────────┐
  │  x₁  ●──┼───────┼──● h₁        │        │            │
  │         │ ╲   ╱ │              │        │            │
  │  x₂  ●──┼───╳───┼──● h₂  ●─────┼────────┼──● ŷ₁      │
  │         │ ╱   ╲ │              │        │            │
  │  x₃  ●──┼───────┼──● h₃        │        │            │
  └─────────┘       └──────────────┘        └────────────┘
   (features)        (representasi          (prediksi)
                      tersembunyi)
```

| Layer | Fungsi |
|-------|--------|
| **Input Layer** | Menerima data masukan (fitur). Jumlah neuron = jumlah fitur |
| **Hidden Layer** | Memproses dan mentransformasi data. Bisa terdiri dari satu atau lebih layer |
| **Output Layer** | Menghasilkan prediksi akhir. Jumlah neuron tergantung jenis tugas |

### 3.3 Fungsi Aktivasi

Fungsi aktivasi memberikan **non-linearitas** pada jaringan, sehingga ANN dapat mempelajari pola yang kompleks.

| Fungsi Aktivasi | Formula | Rentang Output | Kapan Digunakan |
|-----------------|---------|----------------|-----------------|
| **ReLU** | $f(z) = \max(0, z)$ | $[0, +\infty)$ | Hidden layer (paling umum) |
| **Sigmoid** | $f(z) = \frac{1}{1+e^{-z}}$ | $(0, 1)$ | Output klasifikasi biner |
| **Tanh** | $f(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | Hidden layer (alternatif ReLU) |
| **Softmax** | $f(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | $(0, 1)$, total = 1 | Output klasifikasi multi-kelas |
| **Linear** | $f(z) = z$ | $(-\infty, +\infty)$ | Output regresi |

> **Aturan praktis pemilihan aktivasi:**
> - Hidden layer → **ReLU** (cepat, efektif, menghindari *vanishing gradient*)
> - Output regresi → **Tanpa aktivasi** (linear)
> - Output klasifikasi biner → **Sigmoid**
> - Output klasifikasi multi-kelas → **Softmax** (atau tanpa aktivasi jika pakai `CrossEntropyLoss`)

### 3.4 Loss Function (Fungsi Kerugian)

Loss function mengukur seberapa jauh prediksi model dari nilai sebenarnya.

| Jenis Masalah | Loss Function | PyTorch |
|---------------|---------------|---------|
| Regresi | Mean Squared Error (MSE) | `nn.MSELoss()` |
| Regresi | Mean Absolute Error (MAE) | `nn.L1Loss()` |
| Klasifikasi Biner | Binary Cross-Entropy | `nn.BCELoss()` atau `nn.BCEWithLogitsLoss()` |
| Klasifikasi Multi-Kelas | Cross-Entropy | `nn.CrossEntropyLoss()` |

**MSE (Mean Squared Error):**
$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Binary Cross-Entropy:**
$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]$$

**Cross-Entropy (Multi-Class):**
$$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

### 3.5 Optimizer (Pengoptimasi)

Optimizer memperbarui bobot model berdasarkan gradien yang dihitung oleh backpropagation.

| Optimizer | Keterangan | PyTorch |
|-----------|------------|---------|
| **SGD** | Stochastic Gradient Descent — sederhana dan efektif | `torch.optim.SGD()` |
| **Adam** | Adaptive Moment Estimation — cepat konvergen, paling populer | `torch.optim.Adam()` |
| **RMSprop** | Root Mean Square Propagation — baik untuk data non-stasioner | `torch.optim.RMSprop()` |

**Rumus update bobot (Gradient Descent):**
$$w_{baru} = w_{lama} - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}$$

Di mana $\eta$ adalah **learning rate** — seberapa besar langkah update bobot.

### 3.6 Proses Pelatihan ANN

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP (1 Epoch)                      │
│                                                                 │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Forward  │───→│  Hitung   │───→│ Backward │───→│  Update  │  │
│  │  Pass    │    │   Loss    │    │   Pass   │    │  Bobot   │  │
│  │ (ŷ=f(x)) │    │ L(y, ŷ)   │    │ ∂L/∂w    │    │ w=w-η∇L  │  │
│  └──────────┘    └───────────┘    └──────────┘    └──────────┘  │
│       ↑                                                  │      │
│       └──────────────────────────────────────────────────┘      │
│                    Ulangi untuk setiap batch                    │
└─────────────────────────────────────────────────────────────────┘

Ulangi seluruh proses di atas untuk sejumlah EPOCH
```

**Istilah penting:**
- **Epoch**: Satu kali seluruh dataset dilewatkan melalui jaringan
- **Batch**: Subset data yang diproses sekaligus dalam satu iterasi
- **Iteration**: Satu kali update bobot (1 epoch = jumlah data / batch size)
- **Learning Rate**: Tingkat kecepatan model belajar

---

## 4. Persiapan Lingkungan

### 4.1 Instalasi Library

```bash
# Instalasi PyTorch (CPU version)
pip install torch torchvision

# Library tambahan untuk visualisasi dan data
pip install matplotlib numpy scikit-learn seaborn
```

### 4.2 Import Library

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_classification
from sklearn.preprocessing import StandardScaler

# Cek versi PyTorch
print(f"PyTorch Version: {torch.__version__}")

# Cek ketersediaan GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
```

### 4.3 Mengenal `nn.Sequential()`

`nn.Sequential()` adalah cara **paling sederhana** untuk membangun model di PyTorch. Layer-layer disusun secara berurutan, dan data akan mengalir dari layer pertama ke layer terakhir secara otomatis.

```python
# Contoh struktur nn.Sequential
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),   # Layer 1: input → hidden
    nn.ReLU(),                             # Aktivasi ReLU
    nn.Linear(hidden_size, output_size),   # Layer 2: hidden → output
)
```

**Keuntungan `nn.Sequential()`:**
- Kode lebih ringkas dan mudah dibaca
- Tidak perlu menulis fungsi `forward()` secara manual
- Cocok untuk arsitektur *sequential* (layer berurutan)

**Keterbatasan:**
- Tidak bisa membuat arsitektur bercabang (multi-input/output)
- Tidak bisa menambahkan logika khusus di antara layer

---

## 5. Praktikum 1 — Tensor & Operasi Dasar PyTorch

> **Tujuan:** Memahami tensor sebagai struktur data utama di PyTorch dan operasi-operasi dasarnya.

### 5.1 Membuat Tensor

```python
# ============================
# MEMBUAT TENSOR
# ============================

# Dari list Python
tensor_a = torch.tensor([1.0, 2.0, 3.0])
print("Tensor dari list:", tensor_a)
print("Tipe data:", tensor_a.dtype)
print("Shape:", tensor_a.shape)

# Tensor 2D (matriks)
tensor_b = torch.tensor([[1, 2, 3],
                          [4, 5, 6]])
print("\nTensor 2D:\n", tensor_b)
print("Shape:", tensor_b.shape)  # torch.Size([2, 3]) → 2 baris, 3 kolom

# Tensor berisi nol
zeros = torch.zeros(3, 4)
print("\nTensor nol (3x4):\n", zeros)

# Tensor berisi satu
ones = torch.ones(2, 3)
print("\nTensor satu (2x3):\n", ones)

# Tensor acak (distribusi normal)
random_tensor = torch.randn(3, 3)
print("\nTensor acak (3x3):\n", random_tensor)

# Tensor dengan range
range_tensor = torch.arange(0, 10, 2)
print("\nTensor range [0,10) step 2:", range_tensor)
```

### 5.2 Operasi Tensor

```python
# ============================
# OPERASI ARITMATIKA
# ============================

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# Penjumlahan
print("x + y =", x + y)         # tensor([5., 7., 9.])

# Perkalian elemen (element-wise)
print("x * y =", x * y)         # tensor([4., 10., 18.])

# Dot product
print("x · y =", torch.dot(x, y))  # 1*4 + 2*5 + 3*6 = 32

# ============================
# OPERASI MATRIKS
# ============================

A = torch.tensor([[1.0, 2.0],
                   [3.0, 4.0]])

B = torch.tensor([[5.0, 6.0],
                   [7.0, 8.0]])

# Perkalian matriks
C = torch.matmul(A, B)  # atau A @ B
print("\nA × B =\n", C)

# Transpose
print("\nTranspose A:\n", A.T)
```

### 5.3 Autograd — Perhitungan Gradien Otomatis

```python
# ============================
# AUTOGRAD: INTI DARI BACKPROPAGATION
# ============================

# requires_grad=True → PyTorch akan melacak operasi untuk menghitung gradien
x = torch.tensor(3.0, requires_grad=True)

# Misalkan fungsi: y = x² + 2x + 1
y = x**2 + 2*x + 1

# Hitung gradien: dy/dx = 2x + 2
y.backward()

# Untuk x=3: dy/dx = 2(3) + 2 = 8
print(f"x = {x.item()}")
print(f"y = x² + 2x + 1 = {y.item()}")
print(f"dy/dx = 2x + 2 = {x.grad.item()}")  # Output: 8.0

# ----- Contoh dengan beberapa variabel -----

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x_input = torch.tensor(3.0)

# Forward pass: y = w*x + b
y_pred = w * x_input + b

# Loss: L = (y_true - y_pred)²
y_true = torch.tensor(10.0)
loss = (y_true - y_pred)**2

# Backward pass
loss.backward()

print(f"\nw = {w.item()}, b = {b.item()}")
print(f"Prediksi: {y_pred.item()}, Target: {y_true.item()}")
print(f"Loss: {loss.item()}")
print(f"dL/dw = {w.grad.item()}")   # Gradien terhadap w
print(f"dL/db = {b.grad.item()}")   # Gradien terhadap b
```

> **Catatan:**  
> **Autograd** adalah mekanisme PyTorch yang secara otomatis menghitung turunan/gradien. Inilah yang membuat **backpropagation** bekerja — kita tidak perlu menghitung turunan secara manual!

---

## 6. Praktikum 2 — Regresi Linear Sederhana

> **Tujuan:** Membangun model ANN paling sederhana (1 neuron, tanpa hidden layer) untuk mempelajari hubungan linear $y = 2x + 3$.

### 6.1 Membuat Dataset

```python
# ============================
# DATASET: y = 2x + 3 + noise
# ============================

torch.manual_seed(42)  # Agar hasil reproducible

# Membuat 200 data point
n_samples = 200
X = torch.linspace(-5, 5, n_samples).reshape(-1, 1)  # Shape: (200, 1)
y_true = 2 * X + 3 + torch.randn(n_samples, 1) * 0.5  # Tambah noise

# Visualisasi data
plt.figure(figsize=(10, 6))
plt.scatter(X.numpy(), y_true.numpy(), alpha=0.5, s=20, color='steelblue', label='Data')
plt.plot(X.numpy(), (2*X + 3).numpy(), 'r--', linewidth=2, label='y = 2x + 3 (true)')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Dataset Regresi Linear', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 6.2 Split Data & Buat DataLoader

```python
# ============================
# SPLIT DATA: 80% Train, 20% Test
# ============================

# Shuffle indices
indices = torch.randperm(n_samples)
train_size = int(0.8 * n_samples)

X_train = X[indices[:train_size]]
y_train = y_true[indices[:train_size]]
X_test = X[indices[train_size:]]
y_test = y_true[indices[train_size:]]

print(f"Data Training: {X_train.shape[0]} sampel")
print(f"Data Testing : {X_test.shape[0]} sampel")

# Membuat DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### 6.3 Membangun Model

```python
# ============================
# MODEL: Regresi Linear (1 neuron)
# ============================

# Arsitektur:
#   Input (1 fitur) → Linear(1, 1) → Output (1 nilai)
#
# Ini setara dengan: ŷ = w*x + b (1 neuron tanpa aktivasi)

model_regresi = nn.Sequential(
    nn.Linear(in_features=1, out_features=1)  # 1 input → 1 output
)

print("Arsitektur Model:")
print(model_regresi)
print(f"\nJumlah parameter: {sum(p.numel() for p in model_regresi.parameters())}")
# Output: 2 parameter (1 weight + 1 bias)

# Lihat parameter awal (sebelum training)
for name, param in model_regresi.named_parameters():
    print(f"  {name}: {param.data}")
```

### 6.4 Menentukan Loss Function & Optimizer

```python
# ============================
# LOSS FUNCTION & OPTIMIZER
# ============================

# MSE Loss untuk regresi
criterion = nn.MSELoss()

# Optimizer: SGD dengan learning rate 0.01
optimizer = optim.SGD(model_regresi.parameters(), lr=0.01)
```

### 6.5 Training Loop

```python
# ============================
# TRAINING
# ============================

num_epochs = 100
train_losses = []

print("Mulai Training...")
print("-" * 50)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in train_loader:
        # 1. Forward pass: hitung prediksi
        y_pred = model_regresi(X_batch)

        # 2. Hitung loss
        loss = criterion(y_pred, y_batch)

        # 3. Backward pass: hitung gradien
        optimizer.zero_grad()  # Reset gradien ke nol
        loss.backward()        # Hitung gradien

        # 4. Update bobot
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)

    # Print setiap 10 epoch
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | Loss: {avg_loss:.4f}")

print("-" * 50)
print("Training Selesai!")
```

> **PENTING — Urutan langkah training yang WAJIB diingat:**
> 1. **Forward pass** → `y_pred = model(X_batch)`
> 2. **Hitung loss** → `loss = criterion(y_pred, y_batch)`
> 3. **Zero gradients** → `optimizer.zero_grad()`
> 4. **Backward pass** → `loss.backward()`
> 5. **Update weights** → `optimizer.step()`
>
> Urutan ini berlaku untuk SEMUA jenis model di PyTorch!

### 6.6 Visualisasi Training Loss

```python
# ============================
# PLOT TRAINING LOSS
# ============================

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, color='crimson', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Training Loss — Regresi Linear', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 6.7 Evaluasi Model

```python
# ============================
# EVALUASI PADA DATA TEST
# ============================

# Mode evaluasi (menonaktifkan dropout, batch norm, dll.)
model_regresi.eval()

with torch.no_grad():  # Tidak perlu hitung gradien saat evaluasi
    y_pred_test = model_regresi(X_test)
    test_loss = criterion(y_pred_test, y_test)
    print(f"Test Loss (MSE): {test_loss.item():.4f}")

# Lihat bobot yang dipelajari
for name, param in model_regresi.named_parameters():
    print(f"  {name}: {param.data.item():.4f}")
    # Seharusnya mendekati w ≈ 2.0 dan b ≈ 3.0

# ============================
# PLOT HASIL PREDIKSI
# ============================

# Prediksi untuk seluruh range X
model_regresi.eval()
with torch.no_grad():
    X_plot = torch.linspace(-5, 5, 100).reshape(-1, 1)
    y_plot = model_regresi(X_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_test.numpy(), y_test.numpy(), alpha=0.6, s=30,
            color='steelblue', label='Data Test (Aktual)')
plt.plot(X_plot.numpy(), y_plot.numpy(), 'r-', linewidth=2.5,
         label='Prediksi Model')
plt.plot(X_plot.numpy(), (2*X_plot + 3).numpy(), 'g--', linewidth=1.5,
         label='y = 2x + 3 (target)', alpha=0.7)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Hasil Prediksi — Regresi Linear', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

> **Catatan:**  
> Perhatikan bahwa bobot yang dipelajari model (`weight` ≈ 2.0, `bias` ≈ 3.0) mendekati parameter sebenarnya dari fungsi $y = 2x + 3$. Ini menunjukkan bahwa model berhasil "belajar" pola dari data!

---

## 7. Praktikum 3 — Regresi Non-Linear (Multi-Layer)

> **Tujuan:** Membangun ANN dengan hidden layer untuk mempelajari pola non-linear yang tidak bisa ditangani oleh model linear.

### 7.1 Membuat Dataset Non-Linear

```python
# ============================
# DATASET: y = sin(x) + noise
# ============================

torch.manual_seed(42)

n_samples = 500
X = torch.linspace(-2 * np.pi, 2 * np.pi, n_samples).reshape(-1, 1)
y_true = torch.sin(X) + torch.randn(n_samples, 1) * 0.1

# Split data
indices = torch.randperm(n_samples)
train_size = int(0.8 * n_samples)

X_train = X[indices[:train_size]]
y_train = y_true[indices[:train_size]]
X_test = X[indices[train_size:]]
y_test = y_true[indices[train_size:]]

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Visualisasi
plt.figure(figsize=(10, 6))
plt.scatter(X_train.numpy(), y_train.numpy(), alpha=0.3, s=10,
            color='steelblue', label='Train')
plt.scatter(X_test.numpy(), y_test.numpy(), alpha=0.5, s=20,
            color='orange', label='Test')
plt.plot(X.numpy(), torch.sin(X).numpy(), 'r--', linewidth=2,
         label='y = sin(x)')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Dataset Non-Linear: y = sin(x)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 7.2 Membangun Model Multi-Layer

```python
# ============================
# MODEL: ANN dengan Hidden Layers
# ============================

# Arsitektur:
#   Input(1) → Hidden1(64, ReLU) → Hidden2(32, ReLU) → Output(1)
#
#   Layer 1: 1 fitur  → 64 neuron + ReLU
#   Layer 2: 64 neuron → 32 neuron + ReLU
#   Layer 3: 32 neuron → 1 output (tanpa aktivasi, karena regresi)

model_nonlinear = nn.Sequential(
    nn.Linear(1, 64),    # Hidden layer 1
    nn.ReLU(),           # Aktivasi ReLU
    nn.Linear(64, 32),   # Hidden layer 2
    nn.ReLU(),           # Aktivasi ReLU
    nn.Linear(32, 1)     # Output layer (tanpa aktivasi → regresi)
)

print("Arsitektur Model:")
print(model_nonlinear)

# Hitung total parameter
total_params = sum(p.numel() for p in model_nonlinear.parameters())
print(f"\nTotal parameter: {total_params:,}")
# (1*64 + 64) + (64*32 + 32) + (32*1 + 1) = 128 + 2080 + 33 = 2,241
```

> **Mengapa perlu hidden layer?**
>
> Model linear (tanpa hidden layer) hanya bisa memodelkan hubungan linear ($y = wx + b$). Dengan menambahkan hidden layer + fungsi aktivasi non-linear (ReLU), model dapat mempelajari pola non-linear seperti $\sin(x)$, kurva kuadratik, dan pola kompleks lainnya.

### 7.3 Training

```python
# ============================
# LOSS FUNCTION & OPTIMIZER
# ============================

criterion = nn.MSELoss()
optimizer = optim.Adam(model_nonlinear.parameters(), lr=0.001)

# ============================
# TRAINING LOOP
# ============================

num_epochs = 200
train_losses = []
test_losses = []

print("Mulai Training Model Non-Linear...")
print("-" * 60)

for epoch in range(num_epochs):
    # --- Training Phase ---
    model_nonlinear.train()
    epoch_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in train_loader:
        y_pred = model_nonlinear(X_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_train_loss = epoch_loss / num_batches
    train_losses.append(avg_train_loss)

    # --- Evaluation Phase ---
    model_nonlinear.eval()
    with torch.no_grad():
        y_pred_test = model_nonlinear(X_test)
        test_loss = criterion(y_pred_test, y_test)
        test_losses.append(test_loss.item())

    # Print setiap 20 epoch
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Test Loss: {test_loss.item():.6f}")

print("-" * 60)
print("Training Selesai!")
```

### 7.4 Visualisasi Hasil

```python
# ============================
# PLOT 1: Training vs Test Loss
# ============================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot loss
axes[0].plot(train_losses, label='Train Loss', color='crimson', linewidth=2)
axes[0].plot(test_losses, label='Test Loss', color='steelblue',
             linewidth=2, linestyle='--')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('Training vs Test Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# ============================
# PLOT 2: Prediksi vs Aktual
# ============================

model_nonlinear.eval()
with torch.no_grad():
    X_plot = torch.linspace(-2*np.pi, 2*np.pi, 500).reshape(-1, 1)
    y_plot = model_nonlinear(X_plot)

axes[1].scatter(X_test.numpy(), y_test.numpy(), alpha=0.5, s=20,
                color='steelblue', label='Data Test')
axes[1].plot(X_plot.numpy(), y_plot.numpy(), 'r-', linewidth=2.5,
             label='Prediksi Model')
axes[1].plot(X_plot.numpy(), torch.sin(X_plot).numpy(), 'g--',
             linewidth=1.5, alpha=0.7, label='y = sin(x)')
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('y', fontsize=12)
axes[1].set_title('Prediksi Model vs Data Aktual', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 8. Praktikum 4 — Klasifikasi Biner

> **Tujuan:** Membangun ANN untuk klasifikasi biner (2 kelas) menggunakan dataset `make_moons` dari scikit-learn.

### 8.1 Membuat Dataset

```python
# ============================
# DATASET: Make Moons (2 kelas)
# ============================

from sklearn.datasets import make_moons

# Generate data berbentuk bulan sabit
X_np, y_np = make_moons(n_samples=1000, noise=0.2, random_state=42)

# Visualisasi
plt.figure(figsize=(10, 6))
plt.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1],
            c='steelblue', label='Kelas 0', alpha=0.6, s=20)
plt.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1],
            c='coral', label='Kelas 1', alpha=0.6, s=20)
plt.xlabel('Fitur 1', fontsize=12)
plt.ylabel('Fitur 2', fontsize=12)
plt.title('Dataset Make Moons — Klasifikasi Biner', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 8.2 Preprocessing & Split Data

```python
# ============================
# PREPROCESSING
# ============================

# Standardisasi fitur (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_np)

# Split data
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_scaled, y_np, test_size=0.2, random_state=42
)

# Konversi ke tensor PyTorch
X_train = torch.FloatTensor(X_train_np)
y_train = torch.FloatTensor(y_train_np).reshape(-1, 1)  # Shape: (N, 1)
X_test = torch.FloatTensor(X_test_np)
y_test = torch.FloatTensor(y_test_np).reshape(-1, 1)

print(f"X_train shape: {X_train.shape}")  # (800, 2)
print(f"y_train shape: {y_train.shape}")  # (800, 1)
print(f"X_test shape : {X_test.shape}")   # (200, 2)
print(f"y_test shape : {y_test.shape}")   # (200, 1)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### 8.3 Membangun Model

```python
# ============================
# MODEL: Klasifikasi Biner
# ============================

# Arsitektur:
#   Input(2) → Hidden1(32, ReLU) → Hidden2(16, ReLU) → Output(1, Sigmoid)
#
#   - Input: 2 fitur (x₁, x₂)
#   - Hidden 1: 32 neuron dengan ReLU
#   - Hidden 2: 16 neuron dengan ReLU
#   - Output: 1 neuron dengan Sigmoid → probabilitas kelas 1

model_binary = nn.Sequential(
    nn.Linear(2, 32),     # Input → Hidden 1
    nn.ReLU(),
    nn.Linear(32, 16),    # Hidden 1 → Hidden 2
    nn.ReLU(),
    nn.Linear(16, 1),     # Hidden 2 → Output
    nn.Sigmoid()          # Aktivasi Sigmoid → output antara 0-1
)

print("Arsitektur Model:")
print(model_binary)
total_params = sum(p.numel() for p in model_binary.parameters())
print(f"\nTotal parameter: {total_params:,}")
```

### 8.4 Training

```python
# ============================
# LOSS & OPTIMIZER
# ============================

# BCELoss karena output sudah melalui Sigmoid
criterion = nn.BCELoss()
optimizer = optim.Adam(model_binary.parameters(), lr=0.001)

# ============================
# TRAINING LOOP
# ============================

num_epochs = 100
train_losses = []
train_accs = []
test_accs = []

print("Mulai Training Klasifikasi Biner...")
print("-" * 65)

for epoch in range(num_epochs):
    model_binary.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        # Forward pass
        y_pred = model_binary(X_batch)

        # Hitung loss
        loss = criterion(y_pred, y_batch)

        # Backward pass & update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Hitung akurasi (threshold = 0.5)
        predicted = (y_pred >= 0.5).float()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = epoch_loss / len(train_loader)
    train_acc = correct / total * 100
    train_losses.append(avg_loss)
    train_accs.append(train_acc)

    # Evaluasi pada data test
    model_binary.eval()
    with torch.no_grad():
        y_pred_test = model_binary(X_test)
        predicted_test = (y_pred_test >= 0.5).float()
        test_acc = (predicted_test == y_test).sum().item() / y_test.size(0) * 100
        test_accs.append(test_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Test Acc: {test_acc:.1f}%")

print("-" * 65)
print("Training Selesai!")
print(f"\nAkurasi Akhir → Train: {train_accs[-1]:.1f}% | Test: {test_accs[-1]:.1f}%")
```

### 8.5 Visualisasi Hasil

```python
# ============================
# PLOT 1: Loss & Accuracy
# ============================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Loss
axes[0].plot(train_losses, color='crimson', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (BCE)', fontsize=12)
axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(train_accs, label='Train Accuracy', color='crimson', linewidth=2)
axes[1].plot(test_accs, label='Test Accuracy', color='steelblue',
             linewidth=2, linestyle='--')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Train vs Test Accuracy', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================
# PLOT 2: Decision Boundary
# ============================

# Buat grid untuk decision boundary
h = 0.02  # step size
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))

# Prediksi untuk setiap titik di grid
grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
model_binary.eval()
with torch.no_grad():
    Z = model_binary(grid_tensor)
    Z = (Z >= 0.5).float().numpy().reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
plt.scatter(X_test_np[y_test_np == 0, 0], X_test_np[y_test_np == 0, 1],
            c='steelblue', label='Kelas 0', edgecolors='k', s=40, alpha=0.7)
plt.scatter(X_test_np[y_test_np == 1, 0], X_test_np[y_test_np == 1, 1],
            c='coral', label='Kelas 1', edgecolors='k', s=40, alpha=0.7)
plt.xlabel('Fitur 1', fontsize=12)
plt.ylabel('Fitur 2', fontsize=12)
plt.title('Decision Boundary — Klasifikasi Biner',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
```

> **Catatan:**  
> **Decision boundary** menunjukkan bagaimana model membagi ruang fitur menjadi dua area. Garis pemisah yang non-linear menunjukkan bahwa ANN berhasil mempelajari pola yang kompleks berkat hidden layer dan aktivasi ReLU.

### 8.6 Confusion Matrix

```python
# ============================
# CONFUSION MATRIX
# ============================

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

model_binary.eval()
with torch.no_grad():
    y_pred_test = model_binary(X_test)
    y_pred_labels = (y_pred_test >= 0.5).int().numpy().flatten()
    y_true_labels = y_test.int().numpy().flatten()

# Confusion Matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Kelas 0', 'Kelas 1'],
            yticklabels=['Kelas 0', 'Kelas 1'])
plt.xlabel('Prediksi', fontsize=12)
plt.ylabel('Aktual', fontsize=12)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print("=" * 55)
print(classification_report(y_true_labels, y_pred_labels,
                             target_names=['Kelas 0', 'Kelas 1']))
```

---

## 9. Praktikum 5 — Klasifikasi Multi-Kelas (MNIST)

> **Tujuan:** Membangun ANN untuk klasifikasi gambar digit tulisan tangan (0-9) menggunakan dataset MNIST — benchmark klasik dalam deep learning.

### 9.1 Download & Eksplorasi Dataset

```python
# ============================
# DOWNLOAD DATASET MNIST
# ============================

from torchvision import datasets, transforms

# Transform: konversi gambar ke tensor & normalisasi
transform = transforms.Compose([
    transforms.ToTensor(),                  # Konversi ke tensor (0-1)
    transforms.Normalize((0.1307,), (0.3081,))  # Normalisasi (mean, std)
])

# Download dataset
train_dataset = datasets.MNIST(root='./data', train=True,
                                download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False,
                               download=True, transform=transform)

print(f"Jumlah data training: {len(train_dataset)}")
print(f"Jumlah data testing : {len(test_dataset)}")
print(f"Ukuran gambar      : {train_dataset[0][0].shape}")  # (1, 28, 28)
print(f"Jumlah kelas       : 10 (digit 0-9)")
```

### 9.2 Visualisasi Sampel Data

```python
# ============================
# VISUALISASI SAMPEL MNIST
# ============================

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('Sampel Dataset MNIST', fontsize=16, fontweight='bold', y=1.02)

for i, ax in enumerate(axes.flat):
    image, label = train_dataset[i]
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}', fontsize=12, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### 9.3 Membuat DataLoader

```python
# ============================
# DATALOADER
# ============================

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Cek 1 batch
images, labels = next(iter(train_loader))
print(f"Batch images shape: {images.shape}")   # (64, 1, 28, 28)
print(f"Batch labels shape: {labels.shape}")    # (64,)
```

### 9.4 Membangun Model

```python
# ============================
# MODEL: Klasifikasi Multi-Kelas
# ============================

# MNIST: gambar 28x28 = 784 piksel → input layer memiliki 784 neuron
# Output: 10 kelas (digit 0-9)
#
# Arsitektur:
#   Flatten(28×28→784) → Hidden1(256, ReLU) → Dropout(0.2)
#                      → Hidden2(128, ReLU) → Dropout(0.2)
#                      → Output(10)
#
# CATATAN: Kita TIDAK menggunakan Softmax di output karena
#          CrossEntropyLoss di PyTorch sudah termasuk Softmax di dalamnya!

model_mnist = nn.Sequential(
    nn.Flatten(),            # (batch, 1, 28, 28) → (batch, 784)
    nn.Linear(784, 256),     # Hidden layer 1
    nn.ReLU(),
    nn.Dropout(0.2),         # Dropout 20% untuk mengurangi overfitting
    nn.Linear(256, 128),     # Hidden layer 2
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)       # Output layer (10 kelas, TANPA Softmax)
)

print("Arsitektur Model MNIST:")
print(model_mnist)

total_params = sum(p.numel() for p in model_mnist.parameters())
print(f"\nTotal parameter: {total_params:,}")
# (784*256+256) + (256*128+128) + (128*10+10) = 200,960 + 32,896 + 1,290 = 235,146
```

> **Tentang Softmax dan CrossEntropyLoss:**
>
> Di PyTorch, `nn.CrossEntropyLoss()` sudah menggabungkan `nn.LogSoftmax()` dan `nn.NLLLoss()`. Jadi **JANGAN** menambahkan `nn.Softmax()` di output layer jika menggunakan `CrossEntropyLoss`! Menambahkan Softmax ganda akan membuat model tidak belajar dengan benar.

### 9.5 Detail Arsitektur — Aliran Data

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ALIRAN DATA PADA MODEL                          │
│                                                                     │
│  Input Image                                                        │
│  (batch, 1, 28, 28)                                                 │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────┐                                                       │
│  │ Flatten   │  28×28×1 = 784 nilai per gambar                      │
│  └──────────┘                                                       │
│       │ (batch, 784)                                                │
│       ▼                                                             │
│  ┌──────────┐                                                       │
│  │Linear(784│  784 input → 256 output                               │
│  │  → 256)  │  Parameter: 784×256 + 256 = 200,960                   │
│  └──────────┘                                                       │
│       │ (batch, 256)                                                │
│       ▼                                                             │
│  ┌──────────┐                                                       │
│  │   ReLU   │  max(0, z) → menambah non-linearitas                  │
│  └──────────┘                                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────┐                                                       │
│  │Dropout   │  Mematikan 20% neuron secara acak (saat training)     │
│  │  (0.2)   │  → Mencegah overfitting                               │
│  └──────────┘                                                       │
│       │ (batch, 256)                                                │
│       ▼                                                             │
│  ┌──────────┐                                                       │
│  │Linear(256│  256 input → 128 output                               │
│  │  → 128)  │  Parameter: 256×128 + 128 = 32,896                    │
│  └──────────┘                                                       │
│       │ (batch, 128)                                                │
│       ▼                                                             │
│  ┌──────────┐                                                       │
│  │   ReLU   │                                                       │
│  └──────────┘                                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────┐                                                       │
│  │Dropout   │                                                       │
│  │  (0.2)   │                                                       │
│  └──────────┘                                                       │
│       │ (batch, 128)                                                │
│       ▼                                                             │
│  ┌──────────┐                                                       │
│  │Linear(128│  128 input → 10 output (raw scores / logits)          │
│  │  → 10)   │  Parameter: 128×10 + 10 = 1,290                       │
│  └──────────┘                                                       │
│       │ (batch, 10)                                                 │
│       ▼                                                             │
│  Output: 10 logits (skor mentah untuk setiap digit 0-9)             │
│  CrossEntropyLoss akan menerapkan Softmax secara internal           │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.6 Training

```python
# ============================
# LOSS & OPTIMIZER
# ============================

criterion = nn.CrossEntropyLoss()  # Sudah termasuk Softmax!
optimizer = optim.Adam(model_mnist.parameters(), lr=0.001)

# Pindahkan model ke device (GPU jika tersedia)
model_mnist = model_mnist.to(device)

# ============================
# TRAINING LOOP
# ============================

num_epochs = 10
train_losses = []
test_losses = []
train_accs = []
test_accs = []

print("Mulai Training MNIST...")
print("=" * 70)

for epoch in range(num_epochs):
    # ---- TRAINING PHASE ----
    model_mnist.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model_mnist(images)
        loss = criterion(outputs, labels)

        # Backward pass & update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Hitung akurasi
        _, predicted = torch.max(outputs, 1)  # Ambil indeks dengan skor tertinggi
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total * 100
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ---- TESTING PHASE ----
    model_mnist.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model_mnist(images)
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

print("=" * 70)
print(f"Akurasi Akhir → Train: {train_accs[-1]:.2f}% | Test: {test_accs[-1]:.2f}%")
```

### 9.7 Visualisasi Training

```python
# ============================
# PLOT LOSS & ACCURACY
# ============================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Loss
axes[0].plot(range(1, num_epochs+1), train_losses, 'o-',
             color='crimson', linewidth=2, label='Train Loss')
axes[0].plot(range(1, num_epochs+1), test_losses, 's--',
             color='steelblue', linewidth=2, label='Test Loss')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training vs Test Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(range(1, num_epochs+1), train_accs, 'o-',
             color='crimson', linewidth=2, label='Train Accuracy')
axes[1].plot(range(1, num_epochs+1), test_accs, 's--',
             color='steelblue', linewidth=2, label='Test Accuracy')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Training vs Test Accuracy', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 9.8 Visualisasi Prediksi

```python
# ============================
# VISUALISASI PREDIKSI
# ============================

model_mnist.eval()

# Ambil beberapa sampel test
test_images, test_labels = next(iter(test_loader))
test_images_device = test_images.to(device)

with torch.no_grad():
    outputs = model_mnist(test_images_device)
    probabilities = torch.softmax(outputs, dim=1)  # Konversi logits → probabilitas
    _, predictions = torch.max(outputs, 1)

# Plot 10 sampel
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle('Prediksi Model pada Data Test MNIST',
             fontsize=16, fontweight='bold', y=1.02)

for i, ax in enumerate(axes.flat):
    img = test_images[i].squeeze().cpu().numpy()
    true_label = test_labels[i].item()
    pred_label = predictions[i].cpu().item()
    confidence = probabilities[i][pred_label].cpu().item() * 100

    ax.imshow(img, cmap='gray')

    # Warna hijau jika benar, merah jika salah
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f'Pred: {pred_label} ({confidence:.1f}%)\nTrue: {true_label}',
                 fontsize=11, fontweight='bold', color=color)
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### 9.9 Evaluasi Detail per Kelas

```python
# ============================
# EVALUASI DETAIL
# ============================

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Prediksi seluruh data test
model_mnist.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model_mnist(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Classification Report
print("\nClassification Report — MNIST:")
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
plt.title('Confusion Matrix — MNIST', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
```

### 9.10 Menyimpan & Memuat Model

```python
# ============================
# SIMPAN MODEL
# ============================

# Simpan seluruh state model
torch.save(model_mnist.state_dict(), 'model_mnist_ann.pth')
print("(V) Model berhasil disimpan ke 'model_mnist_ann.pth'")

# ============================
# MEMUAT MODEL (untuk penggunaan di masa depan)
# ============================

# Buat arsitektur model yang sama
model_loaded = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)

# Muat bobot
model_loaded.load_state_dict(torch.load('model_mnist_ann.pth'))
model_loaded.eval()
print("(V) Model berhasil dimuat!")

# Verifikasi
with torch.no_grad():
    sample = test_images[0:1]  # Ambil 1 sampel
    output = model_loaded(sample)
    _, pred = torch.max(output, 1)
    print(f"Prediksi model yang dimuat: {pred.item()}")
    print(f"Label sebenarnya         : {test_labels[0].item()}")
```

---

## 10. Ringkasan & Best Practices

### 10.1 Tabel Ringkasan Praktikum

| Praktikum | Tipe Masalah | Input → Output | Aktivasi Output | Loss Function | Metrik |
|-----------|-------------|----------------|-----------------|---------------|--------|
| 2 | Regresi Linear | 1 → 1 | Linear (none) | MSELoss | MSE |
| 3 | Regresi Non-Linear | 1 → 1 | Linear (none) | MSELoss | MSE |
| 4 | Klasifikasi Biner | 2 → 1 | Sigmoid | BCELoss | Accuracy |
| 5 | Klasifikasi Multi-Kelas | 784 → 10 | None* | CrossEntropyLoss | Accuracy |

*\*CrossEntropyLoss sudah mencakup Softmax secara internal*

### 10.2 Pola Umum Kode ANN di PyTorch

```python
# ==========================================
#  TEMPLATE UMUM ANN DENGAN nn.Sequential()
# ==========================================

# 1️. BANGUN MODEL
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size),
    # Tambahkan aktivasi output jika diperlukan
)

# 2️. TENTUKAN LOSS & OPTIMIZER
criterion = nn.MSELoss()  # atau BCELoss(), CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3️ TRAINING LOOP
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        # Forward
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluasi
    model.eval()
    with torch.no_grad():
        # ... hitung metrik pada data test
        pass

# 4️. SIMPAN MODEL
torch.save(model.state_dict(), 'model.pth')
```

### 10.3 Tips & Best Practices

| Topik | Best Practice |
|-------|---------------|
| **Learning Rate** | Mulai dari `0.001` (Adam) atau `0.01` (SGD). Turunkan jika loss *oscillate*. |
| **Batch Size** | Umumnya 32 atau 64. Lebih besar = lebih stabil tapi butuh lebih banyak memori. |
| **Epoch** | Monitor *validation loss*. Hentikan jika sudah tidak turun (*early stopping*). |
| **Aktivasi** | Gunakan **ReLU** untuk hidden layer. Pilih aktivasi output sesuai tipe masalah. |
| **Dropout** | Gunakan 0.2–0.5 di antara hidden layer untuk mengurangi overfitting. |
| **Normalisasi** | Selalu normalisasi/standardisasi input data sebelum memasukkannya ke model. |
| **Seed** | Set `torch.manual_seed(42)` untuk hasil yang reproducible. |
| **GPU** | Gunakan `.to(device)` untuk memindahkan model dan data ke GPU jika tersedia. |

### 10.4 Panduan Pemilihan Arsitektur

```
                    ┌─────────────────┐
                    │ Jenis Masalah?  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌───────────┐  ┌────────────┐
        │ REGRESI  │  │ BINER     │  │ MULTI-     │
        │          │  │ KLASIFI-  │  │ KELAS      │
        └─────┬────┘  │ KASI      │  └──────┬─────┘
              │       └─────┬─────┘         │
              ▼             ▼               ▼
        Output: 1      Output: 1       Output: N
        Aktivasi:      Aktivasi:       Aktivasi:
          None          Sigmoid          None*
        Loss:          Loss:           Loss:
          MSELoss       BCELoss         CrossEntropy
```

---

## 11. Latihan Mandiri

### Latihan 1: Regresi — Prediksi Harga

Bangun model ANN untuk memprediksi harga rumah menggunakan dataset California Housing dari scikit-learn.

```python
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X_np = data.data    # 8 fitur
y_np = data.target  # harga rumah
```

**Tugas:**
1. Lakukan preprocessing (standardisasi)
2. Bangun model `nn.Sequential()` dengan minimal 2 hidden layer
3. Gunakan `MSELoss` dan `Adam` optimizer
4. Latih model selama 50 epoch
5. Visualisasikan training loss
6. Hitung MSE dan MAE pada data test

---

### Latihan 2: Klasifikasi Biner — Tumor

Bangun model ANN untuk mengklasifikasikan tumor sebagai *malignant* atau *benign* menggunakan Breast Cancer dataset.

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X_np = data.data    # 30 fitur
y_np = data.target  # 0 = malignant, 1 = benign
```

**Tugas:**
1. Lakukan standardisasi fitur
2. Bangun model `nn.Sequential()` dengan arsitektur: `30 → 64 → 32 → 1`
3. Gunakan `BCELoss` dan `Adam`
4. Latih model dan catat akurasi per epoch
5. Tampilkan confusion matrix dan classification report

---

### Latihan 3: Klasifikasi Multi-Kelas — Fashion MNIST

Gantikan MNIST biasa dengan Fashion MNIST (gambar baju, sepatu, tas, dll.) dan bandingkan hasilnya.

```python
from torchvision import datasets

train_data = datasets.FashionMNIST(root='./data', train=True,
                                    download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False,
                                   download=True, transform=transform)
```

**Tugas:**
1. Gunakan arsitektur yang sama dengan Praktikum 5
2. Latih model selama 15 epoch
3. Bandingkan akurasi Fashion MNIST vs MNIST biasa
4. Analisis: kelas mana yang paling sulit diklasifikasikan? Mengapa?

---

### Latihan 4: Eksperimen Arsitektur

Menggunakan dataset MNIST, bandingkan performa model dengan arsitektur berbeda:

| Model | Arsitektur |
|-------|-----------|
| A | `784 → 128 → 10` |
| B | `784 → 256 → 128 → 10` |
| C | `784 → 512 → 256 → 128 → 10` |
| D | `784 → 256 → 128 → 10` + Dropout(0.5) |

**Tugas:**
1. Latih semua model dengan hyperparameter yang sama
2. Bandingkan: akurasi, jumlah parameter, waktu training
3. Apakah model yang lebih besar selalu lebih baik? Diskusikan!

---

## 12. Referensi

1. **PyTorch Official Documentation** — [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
2. **PyTorch Tutorials** — [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
3. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press.
4. **LeCun, Y., et al.** (1998). *Gradient-based learning applied to document recognition*. Proceedings of the IEEE.
5. **Kingma, D. P., & Ba, J.** (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015.

---

> **Catatan:** Praktikum ini dirancang sebagai pengantar ANN dasar menggunakan `nn.Sequential()`. Untuk arsitektur yang lebih kompleks dan fleksibel, mahasiswa disarankan mempelajari pendekatan `nn.Module` dengan custom `forward()` method, yang akan dibahas pada praktikum selanjutnya (CNN, RNN, dll.).

---
