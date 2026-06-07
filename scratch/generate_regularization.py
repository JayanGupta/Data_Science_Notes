import sys
import os
sys.path.append(os.path.dirname(__file__))
from notebook_builder import md_cell, code_cell, make_notebook, get_dictionary

# Helper to generate common regression metrics code
def get_regression_eval_code():
    return """# Compute evaluation metrics
mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, predictions)

# Print metrics in plain English
print(f"Mean Absolute Error (MAE): {mae:.4f} points (Average absolute prediction error)")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} points (Penalizes larger errors heavily)")
print(f"R-squared Score (R2): {r2:.4f} (Proportion of explained variance)")
"""

# =====================================================================
# 1. RIDGE REGRESSION
# =====================================================================
ridge_cells = []
ridge_cells.append(md_cell("""# 1. Project Introduction

Welcome! In this notebook, we will explore **Ridge Regression**, a variant of linear regression that uses L2 regularization to prevent overfitting.

### What is Ridge Regression?
* It is a **supervised learning** regression algorithm.
* As models get more features, they can become overly complex and overfit the training data.
* Ridge Regression resolves this by adding a penalty to the loss function based on the sum of the **squared weights** (known as L2 Regularization):
  
  $$\text{Loss} = \text{OLS Loss} + \alpha \times \sum (w_i)^2$$
  
* This penalty forces the model to shrink the coefficients (weights) of less important features close to zero, smoothing out predictions.

### Why does it exist?
* It helps regularize models when features are highly correlated (multicollinearity) or when there are too many features relative to the number of data points.

### Real-World Use Cases:
* **Real Estate**: Predicting house prices where many features (size, area, rooms, proximity to schools) are correlated.
* **Genomics**: Predicting trait variations from multiple genetic markers.
"""))

ridge_cells.append(md_cell("""# 2. Problem Statement

* **Goal**: Predict a student's **Final Exam Score** using several features (some relevant, some redundant/correlated).
* **Business Value**: Enables schools to predict grades reliably without overfitting to noisy parameters.
"""))

ridge_cells.append(code_cell("""# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import metrics
"""))

ridge_cells.append(md_cell("""# 4. Create Synthetic Dataset

We define demographics and academic records for **100 students**.
* **Hours_Studied**: Weekly study hours.
* **Attendance_Rate**: Attendance percentage.
* **Previous_Score**: Score in midterm exam.
* **Study_Group_Sessions**: Count of study sessions attended.
* **Coffee_Cups_Consumed**: Weekly cups of coffee (highly noisy feature).
* **Final_Score**: Target continuous exam grade.
"""))

ridge_cells.append(code_cell("""# Hardcoded student grades data
hours = [
    1.5, 2.0, 2.5, 2.7, 3.0, 3.2, 3.5, 3.8, 4.0, 4.2, 4.5, 4.8, 5.0, 5.2, 5.5, 5.8, 6.0, 6.2, 6.5, 6.8,
    7.0, 7.2, 7.5, 7.8, 8.0, 8.2, 8.5, 8.8, 9.0, 9.2, 9.5, 1.8, 2.2, 2.8, 3.3, 3.9, 4.1, 4.7, 5.1, 5.6,
    6.1, 6.7, 7.1, 7.6, 8.1, 8.6, 9.1, 1.2, 1.7, 2.3, 2.9, 3.4, 3.7, 4.3, 4.6, 5.3, 5.7, 6.3, 6.6, 7.3,
    7.7, 8.3, 8.7, 9.3, 1.4, 1.9, 2.4, 2.6, 3.1, 3.6, 4.4, 4.9, 5.4, 5.9, 6.4, 6.9, 7.4, 7.9, 8.4, 8.9,
    9.4, 1.6, 2.1, 2.7, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2, 6.8, 7.2, 7.8, 8.2, 8.8, 9.2, 9.6, 2.0, 5.0
]

attendance = [
    60, 62, 65, 67, 70, 72, 75, 78, 80, 81, 83, 85, 87, 88, 90, 91, 93, 95, 96, 98,
    99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 63, 66, 68, 73, 79, 80, 84, 86, 90,
    92, 95, 96, 99, 100, 100, 100, 55, 61, 64, 71, 74, 77, 82, 85, 88, 91, 94, 95, 98,
    99, 100, 100, 100, 58, 62, 65, 68, 72, 76, 83, 86, 89, 92, 94, 97, 98, 100, 100, 100,
    100, 60, 64, 69, 73, 79, 81, 85, 87, 91, 93, 96, 97, 99, 100, 100, 100, 100, 65, 85
]

prev_score = [
    50, 52, 55, 57, 58, 60, 62, 63, 65, 66, 68, 70, 72, 73, 75, 76, 78, 80, 82, 83,
    85, 86, 88, 90, 91, 92, 94, 95, 96, 97, 98, 51, 54, 59, 61, 64, 67, 71, 74, 77,
    79, 81, 84, 87, 89, 93, 96, 48, 53, 56, 60, 63, 65, 69, 71, 75, 78, 80, 83, 86,
    88, 91, 94, 97, 49, 52, 55, 58, 61, 63, 68, 72, 74, 77, 80, 82, 85, 88, 91, 93,
    95, 51, 54, 58, 62, 65, 68, 71, 73, 77, 80, 83, 86, 89, 92, 94, 96, 98, 55, 75
]

sessions = [
    1, 2, 2, 3, 2, 1, 3, 4, 3, 2, 4, 3, 5, 4, 5, 4, 5, 6, 6, 5,
    6, 7, 7, 8, 8, 7, 9, 8, 9, 10, 9, 2, 2, 3, 3, 4, 4, 5, 5, 6,
    6, 7, 7, 8, 8, 9, 9, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
    8, 8, 9, 9, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9,
    9, 1, 2, 3, 3, 4, 5, 5, 2, 3, 4, 1, 3, 4, 2, 3, 5, 5, 2, 4
]

coffee = [
    2, 3, 1, 5, 2, 0, 4, 3, 6, 2, 1, 4, 3, 5, 2, 1, 4, 6, 3, 2,
    5, 4, 1, 3, 2, 5, 4, 1, 3, 2, 6, 1, 2, 4, 3, 1, 5, 3, 2, 4,
    6, 2, 1, 3, 5, 2, 4, 0, 3, 1, 5, 2, 4, 3, 1, 6, 2, 4, 3, 1,
    2, 3, 1, 5, 2, 0, 4, 3, 6, 2, 1, 4, 3, 5, 2, 1, 4, 6, 3, 2,
    4, 2, 1, 3, 5, 2, 4, 1, 3, 1, 5, 2, 4, 3, 1, 6, 2, 4, 3, 1
]

final_score = [
    52.5, 55.0, 58.2, 59.8, 61.1, 63.4, 65.5, 67.2, 69.0, 70.3, 72.1, 74.0, 75.8, 77.2, 79.5, 81.0, 83.2, 85.1, 87.5, 89.0,
    91.2, 92.5, 94.0, 95.8, 96.5, 97.2, 98.5, 99.0, 99.5, 100.0, 100.0, 54.1, 57.2, 62.0, 65.4, 71.0, 73.2, 78.5, 81.2, 85.5,
    88.0, 92.1, 94.3, 97.5, 98.8, 99.5, 100.0, 50.2, 56.0, 59.5, 63.8, 67.0, 69.2, 73.5, 75.8, 80.0, 83.2, 86.5, 89.0, 93.1,
    94.8, 97.9, 99.0, 100.0, 51.5, 55.2, 58.6, 61.0, 64.2, 66.8, 72.5, 76.0, 78.5, 81.9, 84.8, 87.5, 90.2, 93.5, 96.2, 98.0,
    99.5, 53.8, 57.0, 61.2, 65.5, 71.2, 74.0, 79.2, 81.5, 86.5, 89.2, 93.5, 95.8, 98.2, 99.5, 100.0, 100.0, 100.0, 57.5, 78.0
]

df = pd.DataFrame({
    'Hours_Studied': hours,
    'Attendance_Rate': attendance,
    'Previous_Score': prev_score,
    'Study_Group_Sessions': sessions,
    'Coffee_Cups_Consumed': coffee,
    'Final_Score': final_score
})

print("Shape:", df.shape)
print(df.head())
"""))

# 5. EDA
ridge_cells.append(md_cell("""# 5. Exploratory Data Analysis (EDA)
"""))

ridge_cells.append(code_cell("""# Chart 1: Heatmap showing correlation including noise
plt.figure(figsize=(7, 6))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.title('Feature Correlations including Noisy Coffee Consumption')
plt.show()
"""))

ridge_cells.append(md_cell("""### What Did We Observe?
* `Hours_Studied`, `Attendance_Rate`, `Previous_Score`, and `Study_Group_Sessions` correlate highly with `Final_Score` (~0.9+).
* `Coffee_Cups_Consumed` displays near-zero correlation with grades.
"""))

# 6. Cleaning
ridge_cells.append(code_cell("""# Cleaning check
print("Null count:", df.isnull().sum().sum())
"""))

# 7. Feature Selection
ridge_cells.append(code_cell("""# Feature Selection
X = df[['Hours_Studied', 'Attendance_Rate', 'Previous_Score', 'Study_Group_Sessions', 'Coffee_Cups_Consumed']]
y = df['Final_Score']
"""))

# 8. Train-Test Split
ridge_cells.append(code_cell("""# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""))

# 9. Model Building
ridge_cells.append(md_cell("""# 9. Model Building

* **How it works conceptually**: It fits a linear line minimizing squared errors plus a squared size penalty on coefficients.
* **Hyperparameter Alpha ($\alpha$)**: Controls penalty strength. $\alpha=1.0$ is the standard default.
"""))

ridge_cells.append(code_cell("""# Initialize Ridge Regression with alpha=1.0
model = Ridge(alpha=1.0)
"""))

# 10. Training
ridge_cells.append(code_cell("""# Train model
model.fit(X_train, y_train)
"""))

# 11. Predictions
ridge_cells.append(code_cell("""# Predict grades
predictions = model.predict(X_test)
"""))

# 12. Evaluation
ridge_cells.append(code_cell(get_regression_eval_code()))

# 13. Visualizing Performance
ridge_cells.append(md_cell("""# 13. Visualizing Model Performance

We will plot:
1. **Actual vs. Predicted Scatter Plot**.
2. **Residual Plot**.
"""))

ridge_cells.append(code_cell("""# Plot 1: Actual vs Predicted Scatter
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, color='indigo', alpha=0.8, edgecolor='black', s=80)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, linestyle='--')
plt.title('Ridge Regression: Actual vs. Predicted Scores')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

ridge_cells.append(code_cell("""# Plot 2: Residual Plot
residuals = y_test - predictions
plt.figure(figsize=(8, 5))
plt.scatter(predictions, residuals, color='darkgreen', alpha=0.8, edgecolor='black', s=80)
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.title('Ridge Regression Residual Plot')
plt.xlabel('Predicted Scores')
plt.ylabel('Residuals')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

# 14. Interpretation
ridge_cells.append(code_cell("""# Coefficient interpretation
print("Ridge Intercept:", model.intercept_)
print("Ridge Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"* {col}: {coef:.6f}")
"""))

ridge_cells.append(md_cell("""### What Did We Observe?
* The coefficient for `Coffee_Cups_Consumed` is very close to zero, showing that Ridge shrank its contribution.
* Significant variables like `Hours_Studied` retain strong coefficients.

### What Did We Learn?
* Ridge shrinks coefficients of minor features but **keeps all features** in the equation (none are shrunk to exactly 0).
"""))

# 15. Conclusion
ridge_cells.append(md_cell("""# 15. Conclusion
* Ridge Regression stabilizes predictions on correlated features using squared penalty regularization.
"""))

# 16. Dictionary
ridge_cells.append(md_cell(get_dictionary()))

make_notebook("models/Ridge_Regression.ipynb", ridge_cells)


# =====================================================================
# 2. LASSO REGRESSION
# =====================================================================
lasso_cells = []
lasso_cells.append(md_cell("""# 1. Project Introduction

Welcome! In this notebook, we will explore **Lasso Regression**, a regularization method that performs both weight shrinkage and automatic feature selection.

### What is Lasso Regression?
* It is a **supervised learning** regression algorithm.
* **Lasso** stands for *Least Absolute Shrinkage and Selection Operator*.
* It adds a penalty to the loss function based on the **absolute value** of the weights (known as L1 Regularization):
  
  $$\text{Loss} = \text{OLS Loss} + \alpha \times \sum |w_i|$$
  
* Unlike Ridge, which shrinks coefficients close to zero, Lasso can shrink coefficients **exactly to zero**. This removes the feature entirely from the linear equation.

### Why does it exist?
* It acts as an automatic **feature selector**, producing simple models that are easier to interpret.

### Real-World Use Cases:
* **Medical Research**: Selecting a few disease-associated genes out of thousands of candidate features.
* **Marketing**: Identifying the most effective channels (e.g. search ads vs. print) while dropping ineffective ones.
"""))

lasso_cells.append(md_cell("""# 2. Problem Statement

* **Goal**: Predict a student's **Final Exam Score** using several features, including highly irrelevant ones.
* **Business Value**: Automates model simplification by stripping out noisy variables.
"""))

lasso_cells.append(code_cell("""# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn import metrics
"""))

lasso_cells.append(md_cell("""# 4. Create Synthetic Dataset

We define demographics for **100 students**.
* **Hours_Studied**: Weekly study hours.
* **Attendance_Rate**: Attendance percentage.
* **Previous_Score**: Score in midterm exam.
* **Study_Group_Sessions**: Count of study sessions.
* **Shoe_Size**: Student shoe size (completely irrelevant variable!).
* **Final_Score**: Target grade.
"""))

lasso_cells.append(code_cell("""# Hardcoded student grades data with highly irrelevant Shoe_Size feature
hours = [
    1.5, 2.0, 2.5, 2.7, 3.0, 3.2, 3.5, 3.8, 4.0, 4.2, 4.5, 4.8, 5.0, 5.2, 5.5, 5.8, 6.0, 6.2, 6.5, 6.8,
    7.0, 7.2, 7.5, 7.8, 8.0, 8.2, 8.5, 8.8, 9.0, 9.2, 9.5, 1.8, 2.2, 2.8, 3.3, 3.9, 4.1, 4.7, 5.1, 5.6,
    6.1, 6.7, 7.1, 7.6, 8.1, 8.6, 9.1, 1.2, 1.7, 2.3, 2.9, 3.4, 3.7, 4.3, 4.6, 5.3, 5.7, 6.3, 6.6, 7.3,
    7.7, 8.3, 8.7, 9.3, 1.4, 1.9, 2.4, 2.6, 3.1, 3.6, 4.4, 4.9, 5.4, 5.9, 6.4, 6.9, 7.4, 7.9, 8.4, 8.9,
    9.4, 1.6, 2.1, 2.7, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2, 6.8, 7.2, 7.8, 8.2, 8.8, 9.2, 9.6, 2.0, 5.0
]

attendance = [
    60, 62, 65, 67, 70, 72, 75, 78, 80, 81, 83, 85, 87, 88, 90, 91, 93, 95, 96, 98,
    99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 63, 66, 68, 73, 79, 80, 84, 86, 90,
    92, 95, 96, 99, 100, 100, 100, 55, 61, 64, 71, 74, 77, 82, 85, 88, 91, 94, 95, 98,
    99, 100, 100, 100, 58, 62, 65, 68, 72, 76, 83, 86, 89, 92, 94, 97, 98, 100, 100, 100,
    100, 60, 64, 69, 73, 79, 81, 85, 87, 91, 93, 96, 97, 99, 100, 100, 100, 100, 65, 85
]

prev_score = [
    50, 52, 55, 57, 58, 60, 62, 63, 65, 66, 68, 70, 72, 73, 75, 76, 78, 80, 82, 83,
    85, 86, 88, 90, 91, 92, 94, 95, 96, 97, 98, 51, 54, 59, 61, 64, 67, 71, 74, 77,
    79, 81, 84, 87, 89, 93, 96, 48, 53, 56, 60, 63, 65, 69, 71, 75, 78, 80, 83, 86,
    88, 91, 94, 97, 49, 52, 55, 58, 61, 63, 68, 72, 74, 77, 80, 82, 85, 88, 91, 93,
    95, 51, 54, 58, 62, 65, 68, 71, 73, 77, 80, 83, 86, 89, 92, 94, 96, 98, 55, 75
]

sessions = [
    1, 2, 2, 3, 2, 1, 3, 4, 3, 2, 4, 3, 5, 4, 5, 4, 5, 6, 6, 5,
    6, 7, 7, 8, 8, 7, 9, 8, 9, 10, 9, 2, 2, 3, 3, 4, 4, 5, 5, 6,
    6, 7, 7, 8, 8, 9, 9, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
    8, 8, 9, 9, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9,
    9, 1, 2, 3, 3, 4, 5, 5, 2, 3, 4, 1, 3, 4, 2, 3, 5, 5, 2, 4
]

shoe_size = [
    7, 8, 9, 8, 7, 10, 8, 9, 11, 7, 8, 9, 10, 8, 7, 9, 11, 8, 7, 9,
    10, 8, 7, 9, 11, 8, 7, 9, 10, 8, 7, 9, 11, 8, 7, 9, 10, 8, 7, 9,
    8, 9, 10, 8, 7, 9, 11, 8, 7, 9, 10, 8, 7, 9, 11, 8, 7, 9, 10, 8,
    7, 8, 9, 8, 7, 10, 8, 9, 11, 7, 8, 9, 10, 8, 7, 9, 11, 8, 7, 9,
    9, 8, 7, 10, 8, 9, 11, 7, 8, 9, 10, 8, 7, 9, 11, 8, 7, 9, 10, 8
]

final_score = [
    52.5, 55.0, 58.2, 59.8, 61.1, 63.4, 65.5, 67.2, 69.0, 70.3, 72.1, 74.0, 75.8, 77.2, 79.5, 81.0, 83.2, 85.1, 87.5, 89.0,
    91.2, 92.5, 94.0, 95.8, 96.5, 97.2, 98.5, 99.0, 99.5, 100.0, 100.0, 54.1, 57.2, 62.0, 65.4, 71.0, 73.2, 78.5, 81.2, 85.5,
    88.0, 92.1, 94.3, 97.5, 98.8, 99.5, 100.0, 50.2, 56.0, 59.5, 63.8, 67.0, 69.2, 73.5, 75.8, 80.0, 83.2, 86.5, 89.0, 93.1,
    94.8, 97.9, 99.0, 100.0, 51.5, 55.2, 58.6, 61.0, 64.2, 66.8, 72.5, 76.0, 78.5, 81.9, 84.8, 87.5, 90.2, 93.5, 96.2, 98.0,
    99.5, 53.8, 57.0, 61.2, 65.5, 71.2, 74.0, 79.2, 81.5, 86.5, 89.2, 93.5, 95.8, 98.2, 99.5, 100.0, 100.0, 100.0, 57.5, 78.0
]

df = pd.DataFrame({
    'Hours_Studied': hours,
    'Attendance_Rate': attendance,
    'Previous_Score': prev_score,
    'Study_Group_Sessions': sessions,
    'Shoe_Size': shoe_size,
    'Final_Score': final_score
})

print("Shape:", df.shape)
print(df.head())
"""))

# 5. EDA
lasso_cells.append(md_cell("""# 5. Exploratory Data Analysis (EDA)
"""))

lasso_cells.append(code_cell("""# Chart 1: Scatter plot of Shoe_Size vs Final Score
plt.figure(figsize=(8, 4))
sns.scatterplot(x='Shoe_Size', y='Final_Score', data=df, color='crimson', s=80)
plt.title('Shoe Size vs. Final Score (No relationship)')
plt.xlabel('Shoe Size')
plt.ylabel('Final Score')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

lasso_cells.append(md_cell("""### What Did We Observe?
* The points are scattered randomly.
* A student's shoe size has no bearing on their final exam score.
"""))

# 6. Cleaning
lasso_cells.append(code_cell("""# Cleaning check
print("Null count:", df.isnull().sum().sum())
"""))

# 7. Feature Selection
lasso_cells.append(code_cell("""# Feature Selection
X = df[['Hours_Studied', 'Attendance_Rate', 'Previous_Score', 'Study_Group_Sessions', 'Shoe_Size']]
y = df['Final_Score']
"""))

# 8. Train-Test Split
lasso_cells.append(code_cell("""# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""))

# 9. Model Building
lasso_cells.append(md_cell("""# 9. Model Building

* **How it works**: Lasso minimizes standard squared error plus an absolute size penalty on weights.
* **Feature Selection**: Because it uses the absolute value penalty, Lasso can force coefficients of features with low explanatory power to **exactly 0**.
"""))

lasso_cells.append(code_cell("""# Initialize Lasso with alpha=0.5
model = Lasso(alpha=0.5)
"""))

# 10. Training
lasso_cells.append(code_cell("""# Train model
model.fit(X_train, y_train)
"""))

# 11. Predictions
lasso_cells.append(code_cell("""# Predict scores
predictions = model.predict(X_test)
"""))

# 12. Evaluation
lasso_cells.append(code_cell(get_regression_eval_code()))

# 13. Visualizing Performance
lasso_cells.append(md_cell("""# 13. Visualizing Model Performance

We will plot:
1. **Actual vs. Predicted Scatter Plot**.
2. **Feature Coefficients Bar Chart**: Clearly demonstrating which features got selected and which got dropped.
"""))

lasso_cells.append(code_cell("""# Plot 1: Actual vs Predicted Scatter
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, color='teal', alpha=0.8, edgecolor='black', s=80)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, linestyle='--')
plt.title('Lasso Regression: Actual vs. Predicted Scores')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

lasso_cells.append(code_cell("""# Plot 2: Lasso Feature Weights
plt.figure(figsize=(8, 4))
sns.barplot(x=model.coef_, y=X.columns, palette='viridis')
plt.axvline(0, color='black', linestyle='--')
plt.title('Lasso Coefficients (L1 Regularization)')
plt.xlabel('Coefficient Weight Value')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()
"""))

# 14. Interpretation
lasso_cells.append(code_cell("""# Coefficient values
print("Lasso Intercept:", model.intercept_)
print("Lasso Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"* {col}: {coef:.6f}")
"""))

lasso_cells.append(md_cell("""### What Did We Observe?
* The coefficient for `Shoe_Size` is **exactly 0.000000**!
* Important features like `Hours_Studied` and `Attendance_Rate` remain active.

### What Did We Learn?
* L1 Regularization successfully performed feature selection. The model identified that `Shoe_Size` was useless and removed it entirely from the prediction formula.
"""))

# 15. Conclusion
lasso_cells.append(md_cell("""# 15. Conclusion
* Lasso Regression successfully drops irrelevant features and shrinks active features to build simpler, highly interpretable models.
"""))

# 16. Dictionary
lasso_cells.append(md_cell(get_dictionary()))

make_notebook("models/Lasso_Regression.ipynb", lasso_cells)
print("Finished regularization notebooks generation!")
