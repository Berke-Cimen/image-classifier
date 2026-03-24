# 🖼️ Image Classification System

PyTorch, timm ve FastAPI ile gelişmiş görüntü sınıflandırma platformu. 20+ hazır model, transfer learning, veri artırma ve GPU hızlandırma desteği ile profesyonel seviyede makine öğrenmesi projeleri geliştirin.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![GPU](https://img.shields.io/badge/GPU-Support-brightgreen.svg)

---

## 📜 İçindekiler

1. Proje Hakkında
2. Özellikler
3. Mimari
4. Hızlı Başlangıç
5. Kurulum
6. Eğitim
7. Model Export
8. API Kullanımı
9. Python API
10. Proje Yapısı
11. Modeller
12. Konfigürasyon
13. Test
14. Veri Formatı
15. SSS

---

## 🎯 Proje Hakkında

Bu proje, **görüntü sınıflandırma** için tasarlanmış komple bir makine öğrenmesi sistemidir. Hem hazır modelleri kullanabilir, hem de kendi özel modellerinizi eğitebilirsiniz.

### Ne İşe Yarar?

| Kullanım Alanı | Açıklama |
|----------------|----------|
| 🏥 Tıp | Radyoloji görüntülerinde hastalık tespiti |
| 🚗 Otomotiv | Araç içi görüntü sınıflandırma |
| 🛒 E-ticaret | Ürün görüntülerini kategorilendirme |
| 🌾 Tarım | Bitki hastalıkları tespiti |
| 📱 Sosyal Medya | İçerik moderasyonu |
| 🔐 Güvenlik | Yüz tanıma ve nesne tespiti |

---

## 🚀 Özellikler

### 🤖 ML Modelleri

| Özellik | Açıklama |
|---------|----------|
| 🧠 20+ Hazır Model | EfficientNet, ResNet, MobileNet, ViT, ConvNeXt |
| 🔄 Transfer Learning | Önceden eğitilmiş ağırlıkları kullan |
| 🏗️ Özel Mimari | CustomCNN ve CustomResNet ile kendi modelinizi tasarlayın |
| 🏭 Model Fabrikası | Tek satır kod ile model oluşturma |

### 🔄 Eğitim Pipeline

| Özellik | Açıklama |
|---------|----------|
| ⚙️ Tam Eğitim Döngüsü | Forward, backward, optimizasyon |
| 📉 LR Schedulers | Cosine, Linear, Step decay |
| 🛑 Early Stopping | Overfitting'i önlemek için |
| 💾 Checkpointing | En iyi modeli otomatik kaydetme |
| 🎨 70+ Veri Artırma | Albumentations ile augmentasyon |

### 📊 Değerlendirme

- Top-1 / Top-5 Accuracy
- Precision, Recall, F1 Score
- Confusion Matrix görselleştirme
- Per-class metrikler

### ⚡ Çıkarım (Inference)

| Özellik | Açıklama |
|---------|----------|
| 🚀 GPU Hızlandırma | CUDA ile 10x daha hızlı |
| 📷 Single Inference | Tek görüntü sınıflandırma |
| 📚 Batch Inference | Yüzlerce görüntüyü toplu işle |
| 📤 ONNX Export | Platform bağımsız deployment |
| 🔥 TorchScript | Production ortamı için |

### 🌐 REST API

| Özellik | Açıklama |
|---------|----------|
| ⚡ FastAPI | Async destekli modern web framework |
| 📖 OpenAPI Docs | Otomatik dokümantasyon |
| 📄 ReDoc | Alternatif dokümantasyon |
| 📎 Multipart Upload | Kolay dosya yükleme |

---

## 🏗️ Mimari

### Sistem Mimarisi

Client → FastAPI → PyTorch (timm + Custom Models)

### Eğitim Pipeline

Veri → DataLoader → Training Loop → Checkpoint

---

## ⚡️ Hızlı Başlangıç

1. git clone ...
2. pip install -r requirements.txt
3. python scripts/train.py --data-dir ./data
4. python run_api.py --model efficientnet_b3
5. curl -X POST http://localhost:8000/api/predict

---

## 📦 Kurulum

### Gereksinimler

| Gereksinim | Versiyon |
|------------|----------|
| 🐍 Python | 3.11+ |
| 🧮 CUDA | 11.8+ (opsiyonel) |

### Kurulum Adımları

git clone https://github.com/Berke-Cimen/image-classifier.git
cd image-classifier
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

---

## 🎓 Eğitim

### Basit Eğitim

python scripts/train.py --data-dir ./data/raw --epochs 100 --batch-size 32 --lr 0.001

### Eğitim Parametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| --data-dir | gerekli | Veri klasörü |
| --epochs | 100 | Epoch sayısı |
| --batch-size | 32 | Batch boyutu |
| --lr | 0.001 | Öğrenme rate |

---

## 🔮 Model Export

### ONNX Formatı

python scripts/export_model.py --checkpoint ./outputs/best_model.pth --format onnx

### TorchScript Formatı

python scripts/export_model.py --checkpoint ./outputs/best_model.pth --format torchscript

---

## 🌐 API Kullanımı

### API Başlatma

python run_api.py --model efficientnet_b3 --num-classes 10 --image-size 300

### API Endpointleri

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | /health | Sağlık kontrolü |
| POST | /api/predict | Tek görüntü sınıflandırma |
| POST | /api/predict-batch | Toplu sınıflandırma |
| GET | /api/model-info | Model bilgileri |

### Tek Görüntü Sınıflandırma

curl -X POST "http://localhost:8000/api/predict" -F "file=@cat.jpg"

---

## 🐍 Python API

### Model Yükleme

from src.models.factory import ModelFactory
from src.inference.predictor import Predictor
from PIL import Image

model = ModelFactory.create(model_name="efficientnet_b3", num_classes=10, pretrained=True)
predictor = Predictor(model=model, class_names=["cat", "dog", "bird"], image_size=300, device="cuda")
result = predictor.predict(Image.open("cat.jpg"))

---

## 📁 Proje Yapısı

image-classifier/
├── configs/                # Konfigürasyon
├── src/
│   ├── data/              # Veri işleme
│   ├── models/            # Model mimarileri
│   ├── training/          # Eğitim loop
│   ├── evaluation/        # Değerlendirme
│   ├── inference/         # Çıkarım
│   └── utils/             # Yardımcılar
├── api/                   # FastAPI
├── scripts/               # Scriptler
├── tests/                 # Testler
├── run_api.py             # API başlatma
└── requirements.txt

---

## 🤖 Modeller

### Model Karşılaştırması

| Model | Parametre | Hız | Doğruluk |
|-------|-----------|-----|----------|
| EfficientNet-B0 | 5.3M | hızlı | iyi
| EfficientNet-B3 | 12M | orta | çok iyi
| EfficientNet-B7 | 66M | yavaş | en iyi
| ResNet-50 | 25.6M | orta | iyi
| MobileNet-V3 | 5.4M | çok hızlı | orta |
| ViT-Small | 22M | orta | çok iyi
| ConvNeXt-S | 50M | orta | çok iyi

### EfficientNet Ailesi

| Model | Giriş | Parametre |
|-------|-------|----------|
| b0 | 224 | 5.3M |
| b1 | 240 | 7.8M |
| b2 | 260 | 9.2M |
| b3 | 300 | 12M |
| b4 | 380 | 19M |
| b5 | 456 | 30M |
| b6 | 528 | 43M |
| b7 | 600 | 66M |

### ResNet Ailesi

ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152

---

## ⚙️ Konfigürasyon

### Model Konfigürasyonu

model:
  name: efficientnet_b3
  pretrained: true
  num_classes: 10
  drop_rate: 0.3
image:
  size: 300
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

### Eğitim Konfigürasyonu

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
optimizer:
  type: adamw
scheduler:
  type: cosine
early_stopping:
  patience: 10
checkpoint:
  save_best: true
  monitor: val_accuracy

---

## 🧪 Test

pytest tests/ -v
pytest tests/ --cov=src --cov-report=html

---

## 📊 Veri Formatı

### Klasör Yapısı

data_dir/
├── class_cat/        # Kedi sınıfı
│   ├── image001.jpg
│   └── image002.jpg
├── class_dog/       # Köpek sınıfı
│   └── ...
└── class_bird/      # Kuş sınıfı
    └── ...

---

## ❓ SSS

### Hangi modeli seçmeliyim?

| İhtiyaç | Önerilen |
|---------|----------|
| Hız gerekli | MobileNet-V3, EfficientNet-B0 |
| Yüksek doğruluk | EfficientNet-B7, ConvNeXt-Large |
| Sınırlı bellek | MobileNet-V3 |
| Dengeli | EfficientNet-B3, ResNet-50 |

### GPU olmadan eğitim yapabilir miyim?

Evet! Model otomatik olarak CPU'ya geçer, ancak çok yavaş olur.

---

## 📜 Lisans

MIT License

---

**👨‍💻 Yapımcı:** Berke Çimen

**📧 İletişim:** berke_cimen@hotmail.com

**🐙 GitHub:** github.com/Berke-Cimen

---

Ipuucu: API dokümantasyonu için http://localhost:8000/docs adresini ziyaret edin!
