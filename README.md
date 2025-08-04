# Iris Flower Classification API

A REST API to predict the Iris flower species using a trained RandomForest model.

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## API Endpoint

- POST `/predict`
  - Input: JSON with features
  - Output: Predicted class (0, 1, 2)
