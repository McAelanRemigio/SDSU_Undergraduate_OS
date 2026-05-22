import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

DATA_PATH = 'cs577_final_dataset.csv'

raw_df = pd.read_csv("filepath")

print('Raw shape:', raw_df.shape)
raw_df.head()

  df = raw_df.drop(columns=[col for col in raw_df.columns if col.startswith('Unnamed')]).copy()


numeric_cols = [
    'TOT_EMP', 'EMP_PRSE', 'H_MEAN', 'A_MEAN', 'MEAN_PRSE',
    'H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75', 'H_PCT90',
    'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90', 'Year'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('#', '208000', regex=False)
            .str.replace('*', '', regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['TOT_EMP', 'A_MEDIAN', 'Year', 'O_GROUP', 'OCC_CODE']).copy()

print('Cleaned shape:', df.shape)
print('Missing values in modeling columns:')
print(df[['TOT_EMP', 'A_MEDIAN', 'Year', 'O_GROUP', 'OCC_CODE']].isna().sum())

df.head()

df[['TOT_EMP', 'A_MEDIAN', 'Year']].describe()

stem_group_map = {
    '15': 'Computer and Mathematical',
    '17': 'Architecture and Engineering',
    '19': 'Life, Physical, and Social Science',
    '29': 'Healthcare Practitioners'
}

df['STEM_GROUP'] = df['OCC_CODE'].astype(str).str[:2].map(stem_group_map)

corr_tot_emp = df['TOT_EMP'].corr(df['A_MEDIAN'])
print(f'Correlation between TOT_EMP and A_MEDIAN: {corr_tot_emp:.4f}')

plt.figure(figsize=(8, 5))
plt.scatter(df['TOT_EMP'], df['A_MEDIAN'], alpha=0.45)
plt.title(f'Total Employment vs Annual Median Wage (r = {corr_tot_emp:.3f})')
plt.xlabel('Total Employment (TOT_EMP)')
plt.ylabel('Annual Median Wage (A_MEDIAN, dollars)')
plt.show()

yearly_salary = df.groupby('Year')['A_MEDIAN'].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.plot(yearly_salary['Year'],yearly_salary['A_MEDIAN'], marker='o')
plt.title('Average Annual Median Wage by Year')
plt.xlabel('Year')
plt.ylabel('Average Annual Median Wage (dollars)')
plt.xticks(sorted(df['Year'].unique()))
plt.show()

yearly_salary

stem_salary = (
    df.groupby('STEM_GROUP')['A_MEDIAN']
    .median()
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(10, 5))
plt.bar(stem_salary['STEM_GROUP'], stem_salary['A_MEDIAN'])
plt.title('Median Wage by STEM Occupation Group')
plt.xlabel('STEM Occupation Group')
plt.ylabel('Median Annual Wage (dollars)')
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
plt.show()

stem_salary

plt.figure(figsize=(8, 5))
plt.hist(df['A_MEDIAN'], bins=30, edgecolor='black')
plt.title('Distribution of Annual Median Wages')
plt.xlabel('Annual Median Wage (dollars)')
plt.ylabel('Number of Occupation-Year Records')
plt.show()

y = df['A_MEDIAN']
constant_prediction = y.mean()
constant_pred_full = np.full(len(y), constant_prediction)

baseline_mse = mean_squared_error(y, constant_pred_full)
baseline_rmse = np.sqrt(baseline_mse)

print(f'Constant prediction: ${constant_prediction:,.2f}')
print(f'Baseline MSE: {baseline_mse:,.2f}')
print(f'Baseline RMSE: ${baseline_rmse:,.2f}')

X_slr = df[['TOT_EMP']]
y = df['A_MEDIAN']

slr_model = LinearRegression()
slr_model.fit(X_slr, y)

slr_intercept = slr_model.intercept_
slr_slope = slr_model.coef_[0]
slr_pred_full = slr_model.predict(X_slr)
slr_mse_full = mean_squared_error(y, slr_pred_full)
slr_rmse_full = np.sqrt(slr_mse_full)

print(f'Model equation: predicted A_MEDIAN = {slr_intercept:,.2f} + ({slr_slope:.6f} * TOT_EMP)')
print(f'SLR MSE: {slr_mse_full:,.2f}')
print(f'SLR RMSE: ${slr_rmse_full:,.2f}')

# Sklearn verification is shown above through the fitted LinearRegression parameters.
plt.figure(figsize=(8, 5))
plt.scatter(df['TOT_EMP'], df['A_MEDIAN'], alpha=0.45, label='Actual')

x_line = np.linspace(df['TOT_EMP'].min(), df['TOT_EMP'].max(), 200)
y_line = slr_intercept + slr_slope * x_line
plt.plot(x_line, y_line, label='SLR Fit')

plt.title('Simple Linear Regression: Employment Predicting Median Wage')
plt.xlabel('Total Employment (TOT_EMP)')
plt.ylabel('Annual Median Wage (A_MEDIAN, dollars)')
plt.legend()
plt.show()

residuals_slr = y - slr_pred_full

plt.figure(figsize=(8, 5))
plt.scatter(slr_pred_full, residuals_slr, alpha=0.45)
plt.axhline(0, linestyle='--')
plt.title('Phase 3 Residual Plot: Simple Linear Regression')
plt.xlabel('Predicted Annual Median Wage (dollars)')
plt.ylabel('Residuals (Actual - Predicted, dollars)')
plt.show()

# phase 3 feature
X = df[['TOT_EMP']]
# target
y = df['A_MEDIAN']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print('Train shape:', X_train.shape)
print('Test shape:', X_test.shape)
print('Train percent:', round(len(X_train) / len(df) * 100, 1))
print('Test percent:', round(len(X_test) / len(df) * 100, 1))

# Scatterplot of data before transformation
df.plot(kind='scatter', x='TOT_EMP', y="A_MEDIAN", title="Feature (TOT_EMP) vs Target (A_MEDIAN)", ylabel="A_MEDIAN ($)")

# Transform data

df_transform = pd.Series()
for index, amount in enumerate(df["TOT_EMP"]):
    df_transform[index] = math.log(amount)

df["TOT_EMP_LOG_TRANSFORM"] = df_transform

# Scatterplot of data after LOG transformation

plot = sns.scatterplot(data=df, x="TOT_EMP_LOG_TRANSFORM", y="A_MEDIAN")
plt.title("Feature (LOG(TOT_EMP)) vs Target (A_MEDIAN)")

def standard_units(x):
    return (x - np.mean(x)) / np.std(x)

def correlation(x, y):
    return np.mean(standard_units(x) * standard_units(y))

# create categorical variable represented as a number from O_GROUP
map = {"major": 0, "detailed":3, "minor":1, "broad":2}
df["O_GROUP_NUM"] = df["O_GROUP"].map(map)

# create interaction term

df['TOT_EMP*O_GROUP_NUM'] = df['TOT_EMP'] * df['O_GROUP_NUM']

#-0.4899324776365431
correlation(df["O_GROUP_NUM"], df["TOT_EMP"])
print("Correlation with each other: -0.4899324776365431")

#-0.04768842433941094
correlation(df["TOT_EMP*O_GROUP_NUM"], df["A_MEDIAN"])
print("Interaction term correlation: -0.04768842433941094")

#retrain model

X_engineered = df[['TOT_EMP', 'TOT_EMP_LOG_TRANSFORM', 'TOT_EMP*O_GROUP_NUM']]

X_train_eng = X_engineered.loc[X_train.index]
X_test_eng  = X_engineered.loc[X_test.index]

model_eng = LinearRegression()
model_eng.fit(X_train_eng, y_train)

y_pred_eng = model_eng.predict(X_test_eng)
rmse_eng = np.sqrt(np.mean((y_test - y_pred_eng) ** 2))
print('Engineered Model Test RMSE:', rmse_eng)

#residual plot for our engineered model using the test set residuals only

residuals_eng = y_test - y_pred_eng
plt.scatter(y_pred_eng, residuals_eng)
plt.axhline(0, color='red')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot: Engineered Model (Test Set)')

plt.scatter(y_test, y_pred_eng)
plt.xlabel('Actual Values (A_MEDIAN)')
plt.ylabel('Predicted Values (A_MEDIAN_hat)')
plt.title('Actual (y) vs. Predicted (yhat)')
min_val = min(min(y_test), min(y_pred_eng))
max_val = max(max(y_test), max(y_pred_eng))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# Retrain Phase 3 Simple Linear Regression with Training Set Only

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
rmse_test = np.sqrt(np.mean((y_test - y_pred_test) ** 2))

# Constant model (baseline)
constant_pred_train = np.full(len(y_train), y_train.mean())
constant_pred_test = np.full(len(y_test), y_train.mean())

constant_mse_train = np.mean((y_train - constant_pred_train)**2)
constant_rmse_train = np.sqrt(constant_mse_train)
constant_rmse_test = np.sqrt(np.mean((y_test - constant_pred_test)**2))

# Phase 3 SLR (already trained as 'model')
phase3_pred_train = model.predict(X_train)
phase3_mse_train = np.mean((y_train - phase3_pred_train)**2)
phase3_rmse_train = np.sqrt(phase3_mse_train)

# Engineered model (already trained as 'model_eng')
engineered_pred_train = model_eng.predict(X_train_eng)
engineered_mse_train = np.mean((y_train - engineered_pred_train)**2)
engineered_rmse_train = np.sqrt(engineered_mse_train)

comparison_table = pd.DataFrame(columns=["Model", "MSE (train)", "RMSE (train)", "RMSE (test)", "Improvement Over Baseline"])

comparison_table.loc[0] = ["Constant Model", constant_mse_train, constant_rmse_train, constant_rmse_test, "N/A"]
comparison_table.loc[1] = ["Phase 3 SRL (retrained)", phase3_mse_train, phase3_rmse_train, "47,277.34", "87.45817"]
comparison_table.loc[2] = ["Engineered Feature Model", engineered_mse_train, engineered_rmse_train, "46,407.12", "957.677817"]

comparison_table.head()
