<div align="center">

# Praktikum Deep Learning

### Laboratorium Sains Data — IKOPIN University

![Semester](https://img.shields.io/badge/Semester-Genap%202025%2F2026-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<p>
Repository resmi untuk kegiatan <strong>Praktikum Deep Learning</strong><br/>
Program Studi S1 Sains Data — Tahun Akademik 2025/2026
</p>

<img src="https://admisi.ikopin.ac.id/assets/images/logo.png" width="300"/>

---

</div>

---

## Deskripsi

Praktikum Deep Learning dirancang untuk memberikan pengalaman **hands-on** kepada mahasiswa dalam memahami, mengimplementasikan, dan mengevaluasi model-model deep learning dari dasar hingga arsitektur state-of-the-art.

> **Mata Kuliah:** Deep Learning (XX4567)
> **SKS Praktikum:** 1 SKS
> **Prasyarat:** Machine Learning, Probabilitas & Statistika, Aljabar Linier

---

## Tim Pengajar

| Peran | Nama | Kontak |
|-------|------|--------|
| Dosen Pengampu | Dr. Ichsan Ibrahim, S.Si., M.Si. | - |
| Asisten Laboratorium | Ryan F F Hakim | - |

---

## Setup & Instalasi

### Opsi Environment

| Opsi | Kelebihan | Kekurangan |
|------|-----------|------------|
| **Google Colab** (Rekomendasi) | Free GPU, tanpa install | Sesi terbatas, internet wajib |
| **Lokal + GPU** | Full kontrol, offline | Perlu GPU NVIDIA |
| **Lokal (CPU only)** | Mudah setup | Training sangat lambat |

### Prasyarat (Instalasi Lokal)

| Software | Versi Minimum | Link Download |
|----------|---------------|---------------|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Git | 2.30+ | [git-scm.com](https://git-scm.com/) |
| CUDA Toolkit | 12.1+ (opsional) | [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads) |
| cuDNN | 8.9+ (opsional) | [developer.nvidia.com](https://developer.nvidia.com/cudnn) |
| Anaconda / Miniconda | Latest | [anaconda.com](https://www.anaconda.com/download) |

### Langkah Instalasi (Google Colab — Cepat)

Cukup buka notebook di Colab dan jalankan cell pertama:

```python
# Cell 1 - Setup (jalankan di awal setiap sesi)
!git clone https://github.com/[username]/praktikum-deep-learning.git
%cd praktikum-deep-learning
!pip install -q -r requirements.txt

import torch
print(f"PyTorch: {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")
```

### Dependencies Utama

```
# Core
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
tensorflow>=2.15.0

# Data & Computation
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.13.0
tensorboard>=2.15.0
wandb>=0.16.0

# NLP
transformers>=4.36.0
tokenizers>=0.15.0
datasets>=2.16.0
nltk>=3.8.0

# Computer Vision
opencv-python>=4.8.0
albumentations>=1.3.0
Pillow>=10.0.0

# Utilities
tqdm>=4.66.0
jupyter>=1.0.0
ipywidgets>=8.1.0
torchsummary>=1.5.1
torchinfo>=1.8.0

# Experiment Tracking
wandb>=0.16.0
mlflow>=2.9.0
```

---

## Jadwal Praktikum

| Pertemuan | Minggu | Topik | Difficulty | Status | Link Modul |
|:---------:|:------:|-------|:----------:|:------:|:---:|
| 1 | 3 | Python Review & NumPy untuk Deep Learning | ⭐ | 🟡 | - |
| 2 | 4 | Perceptron & Neural Network dari Scratch | ⭐⭐ | 🔴 | - |
| 3 | 5 | Framework: PyTorch & TensorFlow | ⭐⭐ | 🔴 | - |
| 4 | 6 | Feedforward Neural Network (FNN) | ⭐⭐ | 🔴 | - |
| 5 | 7 | Convolutional Neural Network (CNN) | ⭐⭐⭐ | 🔴 | - |
| 6 | 8 | **UTS Praktikum** | — | 🔴 | - |
| 7 | 9 | Recurrent Neural Network (RNN/LSTM/GRU) | ⭐⭐⭐ | 🔴 | - |
| 8 | 10 | NLP dengan Deep Learning | ⭐⭐⭐ | 🔴 | - |
| 9 | 11 | Attention & Transformer | ⭐⭐⭐⭐ | 🔴 | - |
| 10 | 12 | Generative Model (VAE & GAN) | ⭐⭐⭐⭐ | 🔴 | - |
| 11 | 13 | Object Detection & Segmentation | ⭐⭐⭐⭐ | 🔴 | - |
| 12 | 14 | Advanced Topics (Diffusion / GNN / RL) | ⭐⭐⭐⭐⭐ | 🔴 | - |

> 🟢 Selesai &nbsp; 🟡 Sedang Berlangsung &nbsp; 🔴 Belum Dimulai

---

## Daftar Modul

<details>
<summary><b>📂 Modul 01 — Python Review & NumPy untuk Deep Learning</b></summary>

### Topik
- Python OOP review (class, inheritance, magic methods)
- NumPy: array operations, broadcasting, vectorization
- Visualisasi data dengan Matplotlib
- Konsep tensor dan operasi tensor

### Learning Objectives
- Mahasiswa mampu melakukan operasi tensor secara efisien
- Mahasiswa memahami konsep broadcasting dan vectorization
- Mahasiswa siap menggunakan framework deep learning

### Tugas
Implementasi operasi matriks untuk forward pass neural network menggunakan NumPy murni

</details>

<details>
<summary><b>📂 Modul 02 — Perceptron & Neural Network dari Scratch</b></summary>

### Topik
- Perceptron: model, learning rule, convergence
- Activation functions (Sigmoid, Tanh, ReLU, Softmax)
- Multi-Layer Perceptron (MLP)
- Backpropagation algorithm dari scratch
- Gradient descent variants

### Learning Objectives
- Mahasiswa memahami mekanisme forward & backward pass
- Mahasiswa mampu mengimplementasikan backpropagation tanpa framework

### Key Equations

$$\hat{y} = \sigma(W^{[L]} \cdot a^{[L-1]} + b^{[L]})$$

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{\partial \mathcal{L}}{\partial z^{[l]}} \cdot (a^{[l-1]})^T$$

### Tugas
Implementasi MLP dari scratch untuk klasifikasi dataset XOR dan MNIST

📁 [`02-perceptron-dan-neural-network-dasar/`](./02-perceptron-dan-neural-network-dasar/)
</details>

<details>
<summary><b>📂 Modul 03 — Framework: PyTorch & TensorFlow</b></summary>

### Topik
- PyTorch: Tensor, Autograd, nn.Module, DataLoader
- TensorFlow/Keras: tf.Tensor, GradientTape, Sequential/Functional API
- Custom Dataset & DataLoader
- Training loop best practices
- Model saving & loading

### Learning Objectives
- Mahasiswa mampu membangun model dengan PyTorch dan TensorFlow
- Mahasiswa memahami automatic differentiation

### Tugas
Re-implementasi MLP Modul 02 menggunakan PyTorch dan TensorFlow

📁 [`03-framework-pytorch-tensorflow/`](./03-framework-pytorch-tensorflow/)
</details>

<details>
<summary><b>📂 Modul 04 — Feedforward Neural Network (FNN)</b></summary>

### Topik
- Arsitektur FNN untuk klasifikasi & regresi
- Loss functions (Cross-Entropy, MSE, Huber)
- Optimizers (SGD, Adam, RMSprop, AdamW)
- Regularization (Dropout, L1/L2, Batch Normalization)
- Learning rate scheduling
- Hyperparameter tuning

### Learning Objectives
- Mahasiswa mampu merancang dan melatih FNN secara end-to-end
- Mahasiswa memahami overfitting dan teknik regularisasi

### Tugas
Klasifikasi tabular dataset (Fashion-MNIST / custom dataset) dengan FNN + eksperimen hyperparameter

📁 [`04-feedforward-neural-network/`](./04-feedforward-neural-network/)
</details>

<details>
<summary><b>📂 Modul 05 — Convolutional Neural Network (CNN)</b></summary>

### Topik
- Operasi konvolusi dan pooling
- Arsitektur CNN klasik: LeNet, AlexNet, VGG, ResNet
- Transfer learning & fine-tuning
- Data augmentation
- Visualisasi: feature maps, Grad-CAM
- Evaluasi: confusion matrix, precision, recall, F1

### Key Architecture

```
Input → [Conv → BN → ReLU → Pool] × N → Flatten → [FC → Dropout] × M → Output
```

### Learning Objectives
- Mahasiswa mampu membangun dan melatih CNN untuk image classification
- Mahasiswa mampu menggunakan pretrained model untuk transfer learning

### Tugas
Image classification pada dataset CIFAR-10 / custom dataset dengan CNN + transfer learning (ResNet/EfficientNet)

📁 [`05-convolutional-neural-network/`](./05-convolutional-neural-network/)
</details>

<details>
<summary><b>📂 Modul 06 — Recurrent Neural Network (RNN/LSTM/GRU)</b></summary>

### Topik
- Sequential data dan konsep recurrence
- Vanilla RNN dan masalah vanishing gradient
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Bidirectional RNN
- Sequence-to-sequence model
- Time series forecasting

### Key Architecture

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(LSTM forget gate)}$$

### Learning Objectives
- Mahasiswa memahami arsitektur recurrent dan gating mechanism
- Mahasiswa mampu menerapkan RNN/LSTM untuk data sekuensial

### Tugas
Time series forecasting (stock price / weather) dan text generation dengan LSTM

📁 [`06-recurrent-neural-network/`](./06-recurrent-neural-network/)
</details>

<details>
<summary><b>📂 Modul 07 — NLP dengan Deep Learning</b></summary>

### Topik
- Text preprocessing (tokenization, padding, vocabulary)
- Word Embeddings (Word2Vec, GloVe, FastText)
- Sentiment Analysis dengan LSTM/GRU
- Text Classification
- Sequence Labeling (NER)

### Learning Objectives
- Mahasiswa mampu memproses data teks untuk deep learning
- Mahasiswa mampu membangun model NLP berbasis deep learning

### Tugas
Sentiment analysis pada dataset review (IMDb / Indonesian tweets) menggunakan LSTM + pre-trained embeddings

📁 [`07-natural-language-processing/`](./07-natural-language-processing/)
</details>

<details>
<summary><b>📂 Modul 08 — Attention & Transformer</b></summary>

### Topik
- Attention mechanism (Bahdanau, Luong)
- Self-attention dan multi-head attention
- Transformer architecture (encoder-decoder)
- Positional encoding
- BERT, GPT overview
- Fine-tuning pretrained transformers (HuggingFace)

### Key Equation

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Learning Objectives
- Mahasiswa memahami mekanisme attention dan arsitektur transformer
- Mahasiswa mampu menggunakan pretrained transformer dari HuggingFace

### Tugas
Text classification menggunakan fine-tuned BERT / GPT-2 text generation

📁 [`08-attention-dan-transformer/`](./08-attention-dan-transformer/)
</details>

<details>
<summary><b>📂 Modul 09 — Generative Model (VAE & GAN)</b></summary>

### Topik
- Autoencoder dan Variational Autoencoder (VAE)
- Generative Adversarial Network (GAN)
- DCGAN, Conditional GAN
- Training stability dan tips
- Evaluation metrics (FID, IS)

### Key Concept

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

### Learning Objectives
- Mahasiswa memahami konsep generative modeling
- Mahasiswa mampu melatih VAE dan GAN untuk image generation

### Tugas
Generate handwritten digits (MNIST) atau anime faces menggunakan DCGAN

📁 [`09-generative-model/`](./09-generative-model/)
</details>

<details>
<summary><b>📂 Modul 10 — Object Detection & Segmentation</b></summary>

### Topik
- Object detection: YOLO, SSD, Faster R-CNN
- Semantic segmentation: FCN, U-Net
- Instance segmentation: Mask R-CNN
- Evaluation: mAP, IoU, Dice coefficient
- Custom dataset annotation (LabelImg, Roboflow)

### Learning Objectives
- Mahasiswa mampu melakukan object detection dan segmentation
- Mahasiswa mampu melatih model pada custom dataset

### Tugas
Object detection pada custom dataset menggunakan YOLOv8 + U-Net semantic segmentation

📁 [`10-object-detection-dan-segmentation/`](./10-object-detection-dan-segmentation/)
</details>

<details>
<summary><b>📂 Modul 11 — Advanced Topics</b></summary>

### Topik (pilih salah satu atau lebih)
- Diffusion Models (DDPM, Stable Diffusion)
- Graph Neural Networks (GCN, GAT)
- Deep Reinforcement Learning (DQN, PPO)
- Speech Recognition & TTS
- Self-Supervised Learning
- Model Deployment (ONNX, TensorRT, Edge devices)

### Learning Objectives
- Mahasiswa memahami perkembangan terkini di bidang deep learning
- Mahasiswa mampu mengeksplorasi topik advanced secara mandiri

### Tugas
Literature review + mini-implementation pada salah satu topik advanced

📁 [`11-advanced-topics/`](./11-advanced-topics/)
</details>

<details>
<summary><b>📂 Modul 12 — Proyek Akhir</b></summary>

### Deskripsi
Proyek kelompok (2-3 orang) yang mengintegrasikan konsep deep learning untuk menyelesaikan permasalahan nyata.

### Contoh Topik Proyek
| No | Topik | Domain |
|----|-------|--------|
| 1 | Real-time Face Mask Detection | Computer Vision |
| 2 | Indonesian Sentiment Analysis | NLP |
| 3 | Music Genre Classification | Audio |
| 4 | Medical Image Segmentation | Healthcare |
| 5 | Chatbot with Transformer | NLP |
| 6 | Image Super Resolution | Computer Vision |
| 7 | Deepfake Detection | Security |
| 8 | Sign Language Recognition | Accessibility |
| 9 | AI-powered Image Captioning | Multimodal |
| 10 | Stock Price Prediction | Finance |

### Deliverables
- Source code + trained model
- Laporan teknis (format paper IEEE)
- Presentasi & live demo
- GitHub repository yang rapi

📁 [`12-proyek-akhir/`](./12-proyek-akhir/)
</details>

---

### ⚠️ Kebijakan AI Tools

| Tool | Kebijakan |
|------|-----------|
| ChatGPT / Copilot | ✅ Boleh untuk debugging & pemahaman konsep |
| Copy-paste output AI | ❌ Dilarang tanpa modifikasi & pemahaman |
| AI-generated report | ❌ Dilarang keras |

> **Prinsip:** AI sebagai **tutor**, bukan **pengganti**. Anda harus bisa menjelaskan setiap baris kode yang Anda tulis.

---

## Tech Stack

<div align="center">

| Kategori | Teknologi |
|----------|-----------|
| Bahasa | Python 3.10+ |
| DL Framework | PyTorch 2.x, TensorFlow 2.x |
| GPU Acceleration | CUDA 12.x, cuDNN 8.x |
| Data Processing | NumPy, Pandas, scikit-learn |
| Computer Vision | OpenCV, torchvision, Albumentations |
| NLP | HuggingFace Transformers, NLTK, tokenizers |
| Audio | torchaudio, Librosa |
| Visualization | Matplotlib, Seaborn, TensorBoard, W&B |
| Experiment Tracking | Weights & Biases, MLflow |
| Notebook | Jupyter, Google Colab |
| Deployment | ONNX, Gradio, Streamlit |

</div>

---

## Referensi & Resources

### Buku Referensi

| Buku | Penulis | Catatan |
|------|---------|---------|
| *Deep Learning* | Goodfellow, Bengio, Courville | Teori comprehensive ([deeplearningbook.org](https://www.deeplearningbook.org/)) |
| *Dive into Deep Learning* | Zhang, Lipton, Li, Smola | Hands-on + code ([d2l.ai](https://d2l.ai/)) |
| *Deep Learning with Python* (2nd Ed.) | François Chollet | Keras/TF focused |
| *Programming PyTorch for Deep Learning* | Ian Pointer | PyTorch focused |

### Online Courses

- [Stanford CS231n — CNNs for Visual Recognition](http://cs231n.stanford.edu/)
- [Stanford CS224n — NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [MIT 6.S191 — Introduction to Deep Learning](http://introtodeeplearning.com/)
- [Fast.ai — Practical Deep Learning](https://course.fast.ai/)
- [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

### Must-Read Papers

| Paper | Tahun | Topik |
|-------|:-----:|-------|
| ImageNet Classification with Deep CNNs (AlexNet) | 2012 | CNN |
| Deep Residual Learning (ResNet) | 2015 | CNN |
| Attention Is All You Need | 2017 | Transformer |
| BERT: Pre-training of Deep Bidirectional Transformers | 2018 | NLP |
| Generative Adversarial Networks | 2014 | GAN |
| Auto-Encoding Variational Bayes | 2013 | VAE |
| You Only Look Once (YOLO) | 2015 | Detection |
| U-Net: Convolutional Networks for Biomedical Segmentation | 2015 | Segmentation |

---

## ❓ FAQ

<details>
<summary><b>Q: Saya tidak punya GPU, apakah bisa mengikuti praktikum?</b></summary>

**A:** Bisa! Gunakan **Google Colab** (gratis, ada GPU T4) atau **Kaggle Notebook** (gratis, GPU P100). Sebagian besar tugas dirancang agar bisa dijalankan di Colab.
</details>

<details>
<summary><b>Q: PyTorch atau TensorFlow, mana yang harus saya fokuskan?</b></summary>

**A:** Praktikum ini mengajarkan keduanya, namun **tugas utama menggunakan PyTorch** karena lebih banyak digunakan di riset. TensorFlow diajarkan sebagai perbandingan.
</details>

<details>
<summary><b>Q: Apakah boleh menggunakan ChatGPT / GitHub Copilot?</b></summary>

**A:** Boleh untuk **memahami konsep** dan **debugging**, tetapi **dilarang** copy-paste langsung tanpa pemahaman. Anda harus bisa menjelaskan setiap baris kode saat ditanya.
</details>

<details>
<summary><b>Q: Bagaimana jika training model terlalu lama?</b></summary>

**A:** Kurangi epoch, gunakan subset data, atau gunakan pretrained model. Diskusikan dengan mentor jika butuh resource tambahan.
</details>

---

<div align="center">

```
"What I cannot create, I do not understand." — Richard Feynman
```

IKOPIN University — Tahun Akademik 2025/2026

</div>
