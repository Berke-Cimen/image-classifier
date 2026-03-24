# Image Classification System

Production-ready image classification system with PyTorch, timm, and FastAPI.

## Features

- **Multiple Architectures**: Support for 20+ pretrained models (EfficientNet, ResNet, MobileNet, ViT, ConvNeXt)
- **Transfer Learning**: Fine-tune pretrained models on your custom dataset
- **Custom Models**: Build custom CNN and ResNet architectures
- **Data Augmentation**: 70+ augmentation strategies via albumentations
- **Training**: Full training loop with LR schedulers, early stopping, checkpoints
- **Evaluation**: Top-1/Top-5 accuracy, precision, recall, F1, confusion matrix
- **Inference**: Fast single/batch inference with GPU acceleration
- **API**: RESTful API with FastAPI, OpenAPI docs
- **Export**: ONNX and TorchScript export

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/image-classifier.git
cd image-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train with pretrained model on custom dataset
python scripts/train.py \
    --data-dir ./data/raw \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001

# Or use a config file
python scripts/train.py --config configs/training_config.yaml
```

### Inference API

```bash
# Start API server
python run_api.py --model efficientnet_b3 --num-classes 10 --image-size 300

# Or with custom class names
python run_api.py --model efficientnet_b3 --num-classes 10 --classes cat dog bird
```

API will be available at `http://localhost:8000`

- API Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Python API

```python
from src.models.factory import ModelFactory
from src.inference.predictor import Predictor
from PIL import Image
import numpy as np

# Load model
model = ModelFactory.create(
    model_name="efficientnet_b3",
    num_classes=10,
    pretrained=True
)

# Create predictor
predictor = Predictor(
    model=model,
    class_names=["class_0", "class_1", ...],
    image_size=300
)

# Predict
image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
result = predictor.predict(image)
print(result)  # {'class_name': 'class_0', 'confidence': 0.95}
```

## Project Structure

```
image-classifier/
├── configs/               # Configuration files
│   ├── model_config.yaml
│   └── training_config.yaml
├── src/
│   ├── data/             # Data pipeline
│   │   ├── dataset.py    # ImageDataset
│   │   ├── transforms.py # Albumentations pipelines
│   │   └── dataloader.py
│   ├── models/           # Model definitions
│   │   ├── factory.py    # timm model factory
│   │   └── architectures.py
│   ├── training/         # Training utilities
│   │   ├── trainer.py
│   │   ├── scheduler.py
│   │   └── callbacks.py
│   ├── evaluation/       # Evaluation metrics
│   │   ├── metrics.py
│   │   └── evaluator.py
│   ├── inference/        # Inference pipeline
│   │   ├── predictor.py
│   │   └── batch_inference.py
│   └── utils/            # Utilities
│       ├── logger.py
│       └── helpers.py
├── api/                  # FastAPI application
│   ├── main.py
│   ├── schemas.py
│   └── routes/
├── scripts/              # Training/evaluation scripts
│   ├── train.py
│   ├── evaluate.py
│   └── export_model.py
├── tests/                # Unit tests
├── run_api.py            # API startup script
├── requirements.txt
└── README.md
```

## Dataset Format

```
data_dir/
├── class_0/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_1/
│   ├── image1.jpg
│   └── ...
└── class_2/
    └── ...
```

## Available Models

### timm Models (Transfer Learning)
- **EfficientNet**: b0, b1, b2, b3, b4, b5, b6, b7
- **ResNet**: 18, 34, 50, 101, 152
- **MobileNet**: v2, v3_large, v3_small
- **ViT**: tiny, small, base, large (patch16)
- **ConvNeXt**: tiny, small, base, large

### Custom Models
- `CustomCNN`: Simple 4-layer CNN
- `CustomResNet`: ResNet-like architecture

## API Endpoints

### Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "model_name": "efficientnet_b3"
}
```

### Single Prediction
```bash
POST /api/predict
Content-Type: multipart/form-data

file: <image_file>
```
Response:
```json
{
  "class_name": "cat",
  "class_index": 0,
  "confidence": 0.95,
  "probabilities": {
    "cat": 0.95,
    "dog": 0.03,
    "bird": 0.02
  }
}
```

### Batch Prediction
```bash
POST /api/predict-batch
Content-Type: multipart/form-data

files: [<image1>, <image2>, ...]
```
Response:
```json
{
  "predictions": [...],
  "total_images": 2,
  "processing_time_ms": 150.5
}
```

### Model Info
```bash
GET /api/model-info
```
Response:
```json
{
  "model_name": "efficientnet_b3",
  "num_classes": 10,
  "class_names": ["cat", "dog", "bird"],
  "image_size": 300
}
```

## Configuration

### Model Config (configs/model_config.yaml)
```yaml
model:
  name: efficientnet_b3
  pretrained: true
  num_classes: 10
  drop_rate: 0.3
  drop_path_rate: 0.2

image:
  size: 300
  channels: 3
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

### Training Config (configs/training_config.yaml)
```yaml
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  warmup_epochs: 5

  optimizer:
    type: adamw
    betas: [0.9, 0.999]
    eps: 1e-8

  scheduler:
    type: cosine
    min_lr: 1e-6
    warmup_type: linear

  augmentation:
    train:
      horizontal_flip: 0.5
      rotate_limit: 15
      scale_limit: 0.1

early_stopping:
  patience: 10
  min_delta: 0.001

checkpoint:
  save_best: true
  save_last: true
  monitor: val_accuracy
  mode: max
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py -v
```

## Export Models

```bash
# Export to ONNX
python scripts/export_model.py \
    --checkpoint ./outputs/best.pth \
    --format onnx

# Export to TorchScript
python scripts/export_model.py \
    --checkpoint ./outputs/best.pth \
    --format torchscript

# Export both
python scripts/export_model.py \
    --checkpoint ./outputs/best.pth \
    --format both
```

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint ./outputs/best.pth \
    --data-dir ./data/raw \
    --output-dir ./eval_results
```

Results include:
- `metrics.json`: All metrics
- `confusion_matrix.png`: Confusion matrix heatmap
- `confusion_matrix_raw.png`: Raw counts
- `per_class_metrics.png`: Per-class precision/recall/F1

## License

MIT License
