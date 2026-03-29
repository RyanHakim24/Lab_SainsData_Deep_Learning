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

| Pertemuan | Minggu | Topik | Status | Link Modul |
|:---------:|:------:|-------| :------:|:---:|
| 1 | 3 | Introduction to PyTorch | 🟢 | [Here](https://github.com/RyanHakim24/Lab_SainsData_Deep_Learning/blob/7919a2afb576763e5888f2b5cdd453a8c1679466/Modul/Praktikum%201%20-%20Pengenalan%20Pytorch.pdf) |
| 2 | 4 | Autograd and Gradient Optimization | 🟡 | - |
| 3 | 5 | - | 🔴 | - |
| 4 | 6 | - | 🔴 | - |
| 5 | 7 | - | 🔴 | - |
| 6 | 8 | - | 🔴 | - |
| 7 | 9 | - | 🔴 | - |
| 8 | 10 | - | 🔴 | - |
| 9 | 11 | - | 🔴 | - |
| 10 | 12 | - | 🔴 | - |
| 11 | 13 | - | 🔴 | - |

> 🟢 Selesai &nbsp; 🟡 Sedang Berlangsung &nbsp; 🔴 Belum Dimulai

---

## Daftar Modul

<details>
<summary><b>📂 Modul 01 — Introduction to PyTorch</b></summary>

### Topik
- Membuat tensor dari berbagai sumber
- Operasi tensor
- Indexing dan Slicing pada tensor
- Tensor pada GPU
</details>

<details>
<summary><b>📂 Modul 02 — Autograd and Gradient Optimization</b></summary>

### Topik
- 
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
