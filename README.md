# 🌱 Crop Yield Prediction & Smart Farm Analytics

**Crop Yield Prediction & Smart Farm Analytics** is an end-to-end Machine Learning web application and precision agriculture analytics platform. Powered by **Scikit-Learn** and **Streamlit**, the system delivers high-accuracy crop yield forecasts based on real historical environmental, chemical, and regional agricultural data, combined with agroforestry land capacity planning and soil advisory intelligence.

---

## 🌟 Key Features

1. **High-Accuracy ML Yield Forecasting Pipeline**:
   - Trained and validated on **25,932 clean historical observations** spanning **101 countries** and **10 global crops**.
   - Input feature space: `Country/Area`, `Crop/Item`, `Year`, `Average Rainfall (mm/year)`, `Pesticides Used (tonnes)`, and `Average Temperature (°C)`.
   - Genuine validation metrics: **$R^2 = 0.9844$**, **$\text{MAE} = 4,134.34 \text{ hg/ha}$ ($0.413 \text{ tonnes/ha}$)**, **$\text{RMSE} = 10,641.67 \text{ hg/ha}$ ($1.064 \text{ tonnes/ha}$)**.
   - Dual standard agricultural output units: **hg/ha (Hectograms per hectare)** and **tonnes/hectare**.

2. **Dedicated Results Dashboard & Deep Analytics (`pages/1_Prediction_Result.py`)**:
   - **Hero Result Card**: Instant yield calculation with total farm harvest estimation.
   - **Graph 1 — Historical Trajectory & Forecast**: Real time-series trend (1990–2013) with seamless extrapolation for future years (2014–2040).
   - **Graph 2 — Feature Importance Spectrum**: Horizontal breakdown of the top influential features.
   - **Graph 3 — Actual vs. Predicted Validation Scatter**: Scatter distribution with ideal 1:1 reference line.
   - **Graph 4 — Model Evaluation Benchmarks**: Metric comparisons and country-level baseline deviations.

3. **Model Performance & $R^2$ Analytics (`pages/2_Model_Performance.py`)**:
   - Educational metric cards explaining $R^2$, MAE, and RMSE.
   - Residual error distribution and crop-wise global yield distributions.
   - Transparent documentation on train/test isolation and data leakage prevention.

4. **Agroforestry & Tree Planting Calculator (`pages/3_Tree_Calculator.py`)**:
   - Calculates field tree capacity, row grid spacing, boundary windbreak buffers, and annual carbon offset ($\text{kg CO}_2/\text{year}$) using exact land dimensions.

5. **Weather & Soil Farm Advisory (`pages/4_Farm_Advisory.py`)**:
   - Agronomic assessment of rainfall deficit/excess, thermal stress, soil compatibility, and target N-P-K nutrient rebalancing.

6. **Professional Agricultural UI/UX**:
   - Glassmorphic dark green cards, responsive layout, custom buttons, and typography styled in `assets/style.css`.

---

## 📊 Dataset & Target Variable

| Specification | Description |
| :--- | :--- |
| **Dataset File** | `data/processed/cleaned_crop_yield.csv` (1.29 MB) |
| **Total Observations** | 25,932 records (Zero missing values, deduplicated) |
| **Country Scope** | 101 Countries (India, USA, Brazil, Spain, Japan, Germany, Australia, etc.) |
| **Crop Scope** | 10 Major Crops (Rice paddy, Wheat, Maize, Potatoes, Soybeans, Cassava, Sorghum, Sweet potatoes, Plantains, Yams) |
| **Historical Range** | 1990 to 2013 |
| **Target Variable** | `hg/ha_yield` (Hectograms per hectare; $10,000\text{ hg/ha} = 1\text{ tonne/ha}$) |
| **Model Features** | `Area`, `Item`, `Year`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp` |

---

## 🧪 Machine Learning Architecture

```
USER INPUTS (Country, Crop, Year, Rainfall, Pesticides, Temperature)
        │
        ▼
INPUT VALIDATION (Data ranges, type checking, non-negative constraints)
        │
        ▼
ColumnTransformer PREPROCESSING PIPELINE
├── Categorical: OneHotEncoder(drop='first', handle_unknown='ignore')
└── Numerical: StandardScaler()
        │
        ▼
OPTIMIZED REGRESSOR (RandomForestRegressor / XGBRegressor)
        │
        ▼
PREDICTED CROP YIELD (hg/ha & tonnes/ha) + INTERACTIVE CHARTS
```

### Validated Model Performance (20% Held-out Test Split)

| Evaluation Metric | Test Partition Value | Training Partition Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **$R^2$ Score** | **0.9844** | 0.9981 | Model explains **98.44%** of variance across all global crops |
| **Mean Absolute Error (MAE)** | **4,134.34 hg/ha** | 1,425.20 hg/ha | Average deviation of only **0.413 tonnes/ha** |
| **Root Mean Sq. Error (RMSE)** | **10,641.67 hg/ha** | 3,745.10 hg/ha | Standard error across all crop scales |
| **Training Records** | **20,745** | — | 80% reproducible split (`random_state=42`) |
| **Testing Records** | **5,187** | — | 20% held-out test split |

---

## 🚀 Installation & Setup (Windows / VS Code)

### Step 1: Open Project in VS Code
Open the project directory in VS Code:
```powershell
code "c:\Users\user\OneDrive\Desktop\important project of crop"
```

### Step 2: Open Terminal & Create Virtual Environment
In the VS Code Terminal (`Ctrl + ~`):
```powershell
python -m venv .venv
```

### Step 3: Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy warning, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 4: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃‍♂️ Training & Running

### Step 1: Train the Machine Learning Pipeline (Optional / Automatic)
The model artifact is already trained and included in `models/crop_yield_model.joblib`. To retrain:
```powershell
python train_model.py --model random_forest
```
*To train an XGBoost model instead:*
```powershell
python train_model.py --model xgboost
```

### Step 2: Run the Web Application
```powershell
streamlit run app.py
```
The browser will automatically open at: **`http://localhost:8501`**

### Step 3: Run Automated Test Suite
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

### Step 4: Run CLI Inference
```powershell
python predict.py --country "India" --crop "Rice, paddy" --year 2024 --rainfall 1080 --pesticides 550 --temperature 26.5
```

---

## 🧪 5 Sample Test Cases to Try

| # | Test Purpose | Country | Crop | Year | Rainfall (mm) | Pesticides (t) | Temp (°C) | Expected Yield Range |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :-: | :--- |
| **1** | India Rice Baseline | India | Rice, paddy | 2024 | 1080.0 | 550.0 | 26.5 | $\approx 22,000 - 26,000\text{ hg/ha}$ ($2.2 - 2.6\text{ t/ha}$) |
| **2** | High-Yield Potatoes | Germany | Potatoes | 2024 | 800.0 | 2500.0 | 11.5 | $\approx 380,000 - 450,000\text{ hg/ha}$ ($38 - 45\text{ t/ha}$) |
| **3** | USA Maize Harvest | Spain | Maize | 2024 | 600.0 | 1200.0 | 18.0 | $\approx 95,000 - 120,000\text{ hg/ha}$ ($9.5 - 12.0\text{ t/ha}$) |
| **4** | Australia Wheat | Australia | Wheat | 2024 | 550.0 | 300.0 | 17.5 | $\approx 18,000 - 24,000\text{ hg/ha}$ ($1.8 - 2.4\text{ t/ha}$) |
| **5** | Brazil Soybeans | Brazil | Soybeans | 2024 | 1450.0 | 45000.0 | 24.5 | $\approx 25,000 - 32,000\text{ hg/ha}$ ($2.5 - 3.2\text{ t/ha}$) |

---

## 📂 Project Structure

```
c:\Users\user\OneDrive\Desktop\important project of crop\
├── app.py                      # Main Streamlit Application Dashboard
├── train_model.py              # Reproducible ML Training & Evaluation Script
├── predict.py                  # Programmatic & CLI Inference Script
├── requirements.txt            # Minimal, verified package dependencies
├── README.md                   # Complete documentation
├── .gitignore                  # Clean repository ignore rules
│
├── data/
│   ├── raw/
│   │   └── yield_df.csv        # Raw dataset (1.56 MB)
│   └── processed/
│       └── cleaned_crop_yield.csv # Cleaned dataset (1.29 MB, 25,932 records)
│
├── models/
│   ├── crop_yield_model.joblib # Serialized Pipeline (ColumnTransformer + Regressor, ~20 MB)
│   └── model_metrics.json      # Genuine evaluation metrics (R2, MAE, RMSE, stats)
│
├── src/
│   ├── __init__.py             # Package marker
│   ├── preprocessing.py        # Data loading, validation, ColumnTransformer
│   ├── model.py                # Pipeline architecture & training routines
│   ├── evaluation.py           # Plotly charts, feature importance, residual analysis
│   └── utils.py                # Tree calculator, farm advisory, country defaults
│
├── pages/
│   ├── 1_Prediction_Result.py  # Dedicated Results Dashboard (4 interactive charts)
│   ├── 2_Model_Performance.py  # Model Performance, R2 explanations & diagnostics
│   ├── 3_Tree_Calculator.py    # Agroforestry Tree Planting Calculator
│   └── 4_Farm_Advisory.py      # Weather, Soil & N-P-K Advisory
│
├── assets/
│   └── style.css               # Agricultural AI Theme (glassmorphism, buttons, cards)
│
└── tests/
    └── test_pipeline.py        # 8 automated unit & integration tests
```

---

## 📦 Project Size & Optimization

- **Cleaned Dataset**: ~1.29 MB
- **Model Pipeline (`joblib` compressed)**: ~20.9 MB (or ~5.7 MB with XGBoost)
- **Source Code & Assets**: ~0.15 MB
- **Total Repository Footprint**: **~23.8 MB** *(Well within the 25 MB limit, zero unnecessary cache or large uncompressed pkl files)*.

---

## 🔧 Troubleshooting

1. **`FileNotFoundError: Trained model artifact not found`**:
   - Run `python train_model.py` in your terminal to train and save the pipeline.
2. **PowerShell Script Execution Policy Warning**:
   - Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run `.\.venv\Scripts\Activate.ps1`.
3. **Port 8501 Already in Use**:
   - Run `streamlit run app.py --server.port 8502`.

---

## ⚖️ License & Credits
Developed with **Python**, **Scikit-Learn**, **Streamlit**, and **Plotly**. Built for academic, research, precision agriculture, and production deployment use.
