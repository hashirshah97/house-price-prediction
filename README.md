# 🏠 Housing Price Prediction — Machine Learning Pipeline

A high-accuracy machine learning project to predict **housing prices** using XGBoost, Random Forest, Gradient Boosting, and a Stacking Ensemble. Built with Python and scikit-learn.

---

## 📁 Project Structure

```
housing-price-ml/
│
├── Housing.csv                  # Dataset (545 rows, 13 features)
├── housing_price_ml.py          # Main ML pipeline script
├── housing_ml_dashboard.png     # Output visualization dashboard
└── README.md                    # This file
```

---

## 📊 Dataset Overview

| Property        | Detail                          |
|-----------------|---------------------------------|
| File            | `Housing.csv`                   |
| Rows            | 545                             |
| Features        | 13                              |
| Target Variable | `price` (continuous)            |
| Missing Values  | None                            |

### Features

| Column             | Type        | Description                            |
|--------------------|-------------|----------------------------------------|
| `price`            | int (target)| House price                            |
| `area`             | int         | Area in square feet                    |
| `bedrooms`         | int         | Number of bedrooms                     |
| `bathrooms`        | int         | Number of bathrooms                    |
| `stories`          | int         | Number of floors                       |
| `mainroad`         | yes/no      | Connected to main road                 |
| `guestroom`        | yes/no      | Has a guest room                       |
| `basement`         | yes/no      | Has a basement                         |
| `hotwaterheating`  | yes/no      | Has hot water heating                  |
| `airconditioning`  | yes/no      | Has air conditioning                   |
| `parking`          | int         | Number of parking spaces               |
| `prefarea`         | yes/no      | Located in preferred area              |
| `furnishingstatus` | categorical | unfurnished / semi-furnished / furnished|

---

## ⚙️ Installation

```bash
pip install xgboost scikit-learn pandas numpy matplotlib
```

---

## 🚀 How to Run

1. Clone or download this project
2. Place `Housing.csv` in the same folder as `housing_price_ml.ipynb`
3. Run the script:



The script will:
- Print a full model comparison table in the terminal
- Save `housing_ml_dashboard.png` to the current directory
- Display a predicted price for a sample house

---

## 🔧 Pipeline Steps

### 1. Preprocessing
- Binary columns (`mainroad`, `guestroom`, etc.) → encoded as `1` / `0`
- `furnishingstatus` → ordinal encoded: `unfurnished=0`, `semi-furnished=1`, `furnished=2`

### 2. Feature Engineering
Four new features are created to boost model accuracy:

| New Feature        | Formula                          | Purpose                    |
|--------------------|----------------------------------|----------------------------|
| `area_per_bedroom` | `area / (bedrooms + 1)`          | Space efficiency           |
| `bath_bed_ratio`   | `bathrooms / (bedrooms + 1)`     | Bathroom-to-bedroom ratio  |
| `total_rooms`      | `bedrooms + bathrooms`           | Overall room count         |
| `luxury_score`     | `ac + hotwater + prefarea + furnishing` | Composite luxury index |

### 3. Train / Test Split
- **80% train / 20% test**, `random_state=42`
- Features scaled with `StandardScaler`

### 4. Models Trained

| Model               | Description                                      |
|---------------------|--------------------------------------------------|
| Random Forest        | 500 trees, ensemble of decision trees           |
| Gradient Boosting    | 500 estimators, lr=0.05, depth=4               |
| XGBoost              | 500 estimators, lr=0.05, depth=5, subsampling  |
| Stacking Ensemble    | RF + GB + XGB stacked with Ridge meta-learner  |

---

## 📈 Results

| Model              | R² Score | CV R² (5-Fold) | MAE       | RMSE      |
|--------------------|----------|----------------|-----------|-----------|
| Random Forest       | 0.5989   | 0.5832         | 1,047,886 | 1,423,946 |
| Gradient Boosting   | 0.6069   | 0.5461         | 1,018,296 | 1,409,575 |
| **XGBoost** ✅      | **0.6468** | **0.5600**   | **974,052** | **1,336,081** |
| Stacking Ensemble   | 0.6145   | 0.5931         | 1,007,780 | 1,395,869 |

> **Best Model: XGBoost** — highest R² score and lowest MAE/RMSE on the test set.


## 📄 License

This project is open-source and free to use for educational and personal purposes.
