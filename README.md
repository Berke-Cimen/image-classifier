# 🖼️ Image Classification System

PyTorch, timm ve FastAPI ile gelişmiş görüntü sınıflandırma platformu. 20+ hazır model, transfer learning, veri artırma ve GPU hızlandırma desteği.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚀 Özellikler

### 🤖 ML Modelleri
- **20+ Hazır Model** - EfficientNet, ResNet, MobileNet, ViT, ConvNeXt
- **Transfer Learning** - Önceden eğitilmiş modelleri kendi verisetinizde fine-tune edin
- **Özel Modeller** - CustomCNN ve CustomResNet mimarileri
- **Model Fabrikası** - Tek satır kodla model oluşturma

### 🔄 Eğitim Pipeline
- **Tam Eğitim Döngüsü** - LR schedulers, early stopping, checkpoints
- **70+ Veri Artırma Stratejisi** - Albumentations kütüphanesi
- **Learning Rate Scheduler** - Cosine, linear, step
- **Callback Sistemi** - Model checkpoint, logging

### 📊 Değerlendirme
- **Top-1 / Top-5 Accuracy**
- **Precision, Recall, F1 Score**
- **Confusion Matrix** görselleştirme
- **Per-class metrikler**

### ⚡ Çıkarım (Inference)
- **GPU Hızlandırma** - CUDA desteği
- **Single / Batch İşleme** - Tek veya toplu görüntü sınıflandırma
- **ONNX Export** - Platform bağımsız dağıtım
- **TorchScript Export** - Production deployment

### 🌐 REST API
- **FastAPI** - Modern async API
- **OpenAPI Docs** - http://localhost:8000/docs
- **ReDoc** - http://localhost:8000/redoc
- **Multipart Upload** - Kolay görüntü yükleme

## 🏗️ Mimari

```
Client → FastAPI → PyTorch (timm + Custom Models)
```

## ⚡ Hızlı Başlangıç

### Kurulum
\`\`\`bash
git clone https://github.com/Berke-Cimen/image-classifier.git
cd image-classifier
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
\`\`\`

### Eğitim
\`\`\`bash
python scripts/train.py --data-dir ./data/raw --epochs 100 --batch-size 32
\`\`\`

### API Başlatma
\`\`\`bash
python run_api.py --model efficientnet_b3 --num-classes 10 --image-size 300
\`\`\`

## 🤖 Kullanılabilir Modeller

| Aile | Modeller |
|------|----------|
| **EfficientNet** | b0, b1, b2, b3, b4, b5, b6, b7 |
| **ResNet** | 18, 34, 50, 101, 152 |
| **MobileNet** | v2, v3_large, v3_small |
| **ViT** | tiny, small, base, large (patch16) |
| **ConvNeXt** | tiny, small, base, large |

## 📡 API Kullanımı

### Tek Görüntü Sınıflandırma
\`\`\`bash
curl -X POST "http://localhost:8000/api/predict" -F "file=@image.jpg"
\`\`\`

### Python API
\`\`\`python
from src.models.factory import ModelFactory
from src.inference.predictor import Predictor

model = ModelFactory.create(model_name="efficientnet_b3", num_classes=10, pretrained=True)
predictor = Predictor(model=model, class_names=["cat", "dog", "bird"], image_size=300)
result = predictor.predict(image)
\`\`\`

## 📜 Lisans

MIT License

---

**Yapımcı:** Berke Çimen
