# ML Model Deployment – CPE393 Homework

This project is part of the CPE393 course, focusing on deploying machine learning models using Flask and Docker.

It is divided into two main parts:

- **Exercise 1–4**: A classification model served with Flask (`app.py`, `train.py`) — runs on port **9000** using Docker image `ml-model`
- **Exercise 5**: A regression model for housing price prediction (`app_regression.py`, `train_regression.py`) — runs on port **9001** using Docker image `regression-app`

---

## File Overview

| File                  | Description                                     |
|-----------------------|-------------------------------------------------|
| `train.py`            | Trains a classification model (Exercises 1–4)   |
| `app.py`              | Flask API for classification                    |
| `train_regression.py` | Trains a regression model using Housing.csv     |
| `app_regression.py`   | Flask API for regression prediction             |
| `model.pkl`           | Trained classification model                    |
| `regression_model.pkl`| Trained regression model                        |
| `Housing.csv`         | Dataset for regression                          |
| `Dockerfile`          | Docker container configuration                  |
| `requirements.txt`    | Python dependencies                             |
| `README.md`           | Project documentation (this file)              |

---

## Setup Instructions (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train models

Train classification model:

```bash
python train.py
```

Train regression model:

```bash
python train_regression.py
```

---

## Run Flask APIs

### Classification API

```bash
python app.py
```

Runs at:

```
http://127.0.0.1:9000
```

### Regression API

```bash
python app_regression.py
```

Runs at:

```
http://127.0.0.1:9001
```

---

## Docker Instructions

### Classification (Docker on port 9000, image: ml-model)

```bash
docker build -t ml-model .
docker run -p 9000:9000 ml-model
```

### Regression (Docker on port 9001, image: regression-app)

```bash
docker build -t regression-app -f Dockerfile .
docker run -p 9001:9001 regression-app
```

---

## Sample API Requests

### Classification (http://127.0.0.1:9000)

#### Single Input

```bash
curl -X POST http://127.0.0.1:9000/predict \
     -H "Content-Type: application/json" \
     -d '{"input_features": [5.1, 3.5, 1.4, 0.2]}'
```

#### Multiple Inputs

```bash
curl -X POST http://127.0.0.1:9000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]}'
```

---

### Regression (http://127.0.0.1:9001)

```bash
curl -X POST http://127.0.0.1:9001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_features": [
      {
        "area": 6000,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "yes",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": 2,
        "prefarea": "yes",
        "furnishingstatus": "unfurnished"
      }
    ]
  }'
```

---

## Health Check

### Classification

```bash
curl http://localhost:9000/health
```

```json
{
  "status": "ok"
}
```

---

## Author

64070503468 Fasai Kumarnjan
