# SecureID Document Classifier

An end-to-end machine learning system for classifying identity document images using a lightweight MobileNet-based image classifier.

The system exposes a FastAPI endpoint for image upload and prediction, and is fully containerized using Docker and Docker Compose.

## Features

- Passport vs Driver License vs Other classification
- Lightweight MobileNet model
- FastAPI REST API
- Swagger UI documentation
- Dockerized deployment
- Validation accuracy tracking
- End-to-end inference pipeline

## Project Structure

```text
secureid-classifier/
│
├── app/
│   └── main.py                # FastAPI inference API
│
├── training/
│   └── train.py               # Model training script
│
├── data/
│   ├── train/
│   └── val/
│
├── model/
│   └── model.pth              # Trained model weights
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

## Installation

### Clone Repository

```bash
git clone <your-repository-url>
cd secureid-classifier
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
.\venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Model Training

Run the training script:

```bash
python training/train.py
```

---

## Run FastAPI Locally

```bash
uvicorn app.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## Docker Deployment

### Build and Run Using Docker Compose

```bash
docker compose up
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```
## Dataset

A curated dataset of publicly available passport, driver license, and non-document images was assembled for lightweight local training and experimentation.

Due to the large storage and compute requirements of full-scale public identity document benchmarks such as MIDV-2020, a curated lightweight dataset was used to enable efficient local experimentation, rapid iteration, and CPU-friendly training while still demonstrating the complete machine learning pipeline.

Dataset structure:

```text
data/
├── train/
│   ├── passport/
│   ├── drivers_license/
│   └── other/
│
└── val/
    ├── passport/
    ├── drivers_license/
    └── other/
```

---

## Model Details

- Backbone: MobileNetV2
- Framework: PyTorch
- Input Size: 224x224
- Classes:
  - passport
  - drivers_license
  - other

---

## Validation Performance

The model achieved a peak validation accuracy of approximately 90% on the held-out validation dataset.

Training logs include:
- Epoch loss
- Validation accuracy tracking

## API Endpoint

### POST `/v1/classify`

Upload an image file for document classification.

### Example Response

```json
{
  "predicted_class": "passport",
  "confidence": 92.45,
  "processing_time": 0.18
}
```
## Future Improvements

- Larger and more diverse training dataset
- Advanced data augmentation
- Improved validation pipeline
- OCR integration for document field extraction
- Cloud deployment support
- CI/CD integration
