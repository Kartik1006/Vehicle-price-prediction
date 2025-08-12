# Vehicle Price Prediction

A machine learning project to predict used vehicle prices based on key features such as year, mileage, cylinders, and optionally categorical features like make, model, and fuel type.

---

## Project Structure
```
app.py # Streamlit / Flask app for deployment
vehicle_price_model.pkl # Trained machine learning model
requirements.txt # Python dependencies
dataset.csv # Sample dataset (raw)
README.md # Project documentation
```

---

## Problem Statement

Used car prices are influenced by various factors such as manufacturing year, mileage, engine specifications, brand, and model.  
This project aims to develop a machine learning model that can predict the resale price of a car based on these features.

---

## Tech Stack

- **Python 3.x**
- **LightGBM** (Regressor)
- **pandas**, **numpy** (Data handling)
- **scikit-learn** (Utilities)
- **joblib** (Model persistence)
- **Streamlit** (Interactive UI)
- *(Optional: Flask for API deployment)*

---

## Project Features

- Predicts resale price of vehicles using simple numeric inputs (`year`, `mileage`, `cylinders`).
- Can be extended to include categorical features (`make`, `model`, etc.).
- Web interface using **Streamlit** for user-friendly interaction.

---

## How to Use

### 1. Clone the Repository

```
git clone https://github.com/Kartik1006/Vehicle-price-prediction.git
cd vehicle-price-prediction
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```
### 3. Run Streamlit App
```
streamlit run app.py
```
---
## Input Features

| Feature      | Type   | Description                      |
|--------------|--------|----------------------------------|
| year         | int    | Year of manufacture              |
| mileage      | int    | Mileage in kilometers or miles   |
| cylinders    | int    | Number of cylinders              |
| *(Optional)* | object | Make, Model, etc. (future scope) |


Output
Predicted price of the vehicle in dollars (or your preferred currency).

---

### Future Improvements
- Incorporate more features (make, model, fuel type, etc.) via label encoding.
- Extend to API deployment (Flask/FastAPI) for production use.

### Known Issues
- Predictions using only three features may be less accurate.
- Streamlit must be run using streamlit run app.py and not with python app.py.

Requirements
```
streamlit
scikit-learn
pandas
numpy
joblib
lightgbm
```
### Contributions
Contributions are welcome. Please fork the repository and submit a pull request.
