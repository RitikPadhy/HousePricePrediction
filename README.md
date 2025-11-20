# House Price Prediction

## Introduction
This project focuses on predicting house prices using a regression-based machine learning approach. The dataset contains 81 features and 1460 entries, with `SalePrice` as the target variable. The goal is to preprocess the data, train multiple regression models, and select the best-performing model to predict house prices.

---

## Dataset
The dataset includes four files:  
- `train.csv` – Training set with 1460 entries and 81 columns  
- `test.csv` – Testing set  
- `data_description.txt` – Column descriptions  
- `sample_submission.csv` – Submission template  

The prediction target is `SalePrice`, and all analysis is performed on `train.csv`.

---

## Data Preprocessing

### Step 1: Handling Missing Values
- Checked for null values using `dataset.isnull()`.  
- Dropped columns with more than 50% null values.  
- Filled numerical columns with their respective means.  
- Filled categorical columns with their respective mode.  
- Verified that all null values were handled.

### Step 2: Data Exploration & Visualization
- Explored dataset shape, column types, and statistical summary.  
- Plotted distribution of `SalePrice` and analyzed skewness.  
- Computed correlation matrix and visualized high correlation features with heatmaps.

### Step 3: Encoding & Scaling
- Encoded categorical variables using `pd.get_dummies()`.  
- Split data into training (80%) and testing (20%) sets.  
- Standardized features using `StandardScaler`.

---

## Model Building
Three regression models were trained and evaluated:

1. **Multiple Linear Regression**  
2. **Random Forest Regressor**  
3. **XGBoost Regressor**  

Performance was measured using the `r2_score`.  

**Results:**
- Multiple Linear Regression: -3.54e+21 (Poor)  
- Random Forest Regressor: 0.838 (Best)  
- XGBoost Regressor: 0.811 (Second Best)  

**Selected Model:** Random Forest Regressor

---

## Hyperparameter Tuning
- Used `RandomizedSearchCV` for optimizing hyperparameters of Random Forest.  
- 50 iterations, 5-fold cross-validation, using all processors.  

**Improved Random Forest Performance:** 0.843 (R² Score)

---

## Conclusion
- Random Forest Regressor is the best model for predicting house prices in this dataset.  
- Hyperparameter tuning improved performance slightly.  
- This pipeline can be used for reproducible house price prediction and further experimentation.

---

## Usage
1. Load the dataset in a Python environment (e.g., Google Colab).  
2. Run preprocessing scripts to handle missing data and encode categorical variables.  
3. Train regression models and evaluate performance.  
4. Optionally, perform hyperparameter tuning for Random Forest Regressor.
