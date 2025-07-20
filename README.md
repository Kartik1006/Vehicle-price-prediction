# 🚗 Vehicle Price Prediction

A machine learning project to predict used vehicle prices based on key features such as year, mileage, cylinders, and optionally categorical features like make, model, fuel type, etc.

---

## 📂 Project Structure

```
🔹 app.py                  # Streamlit / Flask app for deployment
🔹 vehicle_price_model.pkl  # Trained machine learning model
🔹 requirements.txt         # Python dependencies
🔹 dataset.csv               # Sample dataset (raw)
🔹 README.md                # Project documentation
```

---

## 💡 Problem Statement

Used car prices are influenced by various factors like manufacturing year, mileage, engine specs, brand, and model. This project aims to build a machine learning model that can reasonably predict the resale price of a car given these features.

---

## ⚙️ Tech Stack

- **Python 3.x**
- **LightGBM** (Regressor)
- **pandas**, **numpy** (Data handling)
- **scikit-learn** (Utilities)
- **joblib** (Model persistence)
- **Streamlit** (For interactive UI)
- *(Optionally Flask if deploying via API)*

---

## 🚀 Project Features

- Predicts resale price of vehicles using simple numeric inputs (`year`, `mileage`, `cylinders`).
- Can be extended to include categorical features (`make`, `model`, etc.).
- Web interface using **Streamlit** for user-friendly interaction.

---

## 📝 How to Use

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Kartik1006/Vehicle-price-prediction.git
cd vehicle-price-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🧑‍💻 Input Features

| Feature      | Type   | Description                      |
| ------------ | ------ | -------------------------------- |
| year         | int    | Year of manufacture              |
| mileage      | int    | Mileage in km/miles              |
| cylinders    | int    | Number of cylinders              |
| *(Optional)* | object | Make, Model, etc. (future scope) |

---

## 📈 Output

Predicted price of the vehicle in dollars (or your preferred currency).

---

## 📷 Sample Streamlit UI

```
+-----------------------+
| Year: [2018]          |
| Mileage: [45000]      |
| Cylinders: [4]        |
| [ Predict Price ]     |
+-----------------------+
```

Output:\
**Predicted Vehicle Price: \$13,850.00**

---

## 🔮 Future Improvements

- Incorporate more features (make, model, fuel type, etc.) via label encoding.
- Extend to API deployment (Flask/FastAPI) for production use.

---

## 🏗️ Known Issues

- Predictions with only 3 features are limited and may lack accuracy.
- Streamlit must be run via `streamlit run app.py`, not as `python app.py`.

---

## ⚒️ Requirements

```
streamlit
scikit-learn
pandas
numpy
joblib
lightgbm
```

---

## 🤝 Contributions

Contributions are welcome! Please fork the repository and submit a pull request.

---

## 📜 License

[MIT License](https://opensource.org/licenses/MIT)
