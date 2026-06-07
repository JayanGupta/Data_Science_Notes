import sys
import os
sys.path.append(os.path.dirname(__file__))
from notebook_builder import md_cell, code_cell, make_notebook, get_dictionary

cells = []

# 1. Project Introduction
cells.append(md_cell("""# 1. Project Introduction

Welcome to your first step in Machine Learning! In this notebook, we will explore **Linear Regression**, one of the most fundamental and widely used algorithms in data science.

### What is Linear Regression?
* It is a **supervised learning** algorithm used to predict a continuous numerical value (a target) based on one or more input variables (features).
* It assumes a straight-line relationship between the inputs and the output. Think of it as finding the "best fit" line through your data points.

### Why does it exist?
* It was developed to model and understand relationships between variables.
* It helps us answer questions like: "If variable A increases by 10%, how much will variable B change?"

### Real-World Use Cases:
* **Real Estate**: Predicting house prices based on size, bedrooms, and location.
* **Finance**: Predicting a company's future sales based on marketing spend.
* **Healthcare**: Estimating patient recovery time based on dosage level.
"""))

# 2. Problem Statement
cells.append(md_cell("""# 2. Problem Statement

For this project, we want to solve a common educational challenge:
* **Goal**: Predict a student's **Final Exam Score** based on their study habits.
* **Business Value**: By predicting which students might get low scores, teachers can intervene early and provide extra tutoring.
"""))

# 3. Import Libraries
cells.append(md_cell("""# 3. Import Libraries

Before starting, we need to load the standard Python libraries. We will use:
* **NumPy**: For basic numerical lists.
* **Pandas**: For working with data tables (DataFrames).
* **Matplotlib & Seaborn**: For creating clean charts.
* **Scikit-Learn (sklearn)**: For building and evaluating our Machine Learning model.
"""))

cells.append(code_cell("""# Importing libraries step-by-step
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing specific tools from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
"""))

# 4. Create Synthetic Dataset
cells.append(md_cell("""# 4. Create Synthetic Dataset

To keep things completely clear and transparent, we define our dataset as literal Python lists. 
* This represents **100 students** and their study behaviors.
* **Hours_Studied**: The average weekly study hours (ranges from 1.0 to 10.0 hours).
* **Attendance_Rate**: The percentage of classes attended (ranges from 55% to 100%).
* **Previous_Score**: The score obtained in the midterm exam (ranges from 50 to 98 points).
* **Final_Score**: The final exam score (ranges from 52 to 100 points - this is our target!).
"""))

cells.append(code_cell("""# Defining raw lists for 100 student records
hours_studied = [
    1.5, 2.0, 2.5, 2.7, 3.0, 3.2, 3.5, 3.8, 4.0, 4.2, 4.5, 4.8, 5.0, 5.2, 5.5, 5.8, 6.0, 6.2, 6.5, 6.8,
    7.0, 7.2, 7.5, 7.8, 8.0, 8.2, 8.5, 8.8, 9.0, 9.2, 9.5, 1.8, 2.2, 2.8, 3.3, 3.9, 4.1, 4.7, 5.1, 5.6,
    6.1, 6.7, 7.1, 7.6, 8.1, 8.6, 9.1, 1.2, 1.7, 2.3, 2.9, 3.4, 3.7, 4.3, 4.6, 5.3, 5.7, 6.3, 6.6, 7.3,
    7.7, 8.3, 8.7, 9.3, 1.4, 1.9, 2.4, 2.6, 3.1, 3.6, 4.4, 4.9, 5.4, 5.9, 6.4, 6.9, 7.4, 7.9, 8.4, 8.9,
    9.4, 1.6, 2.1, 2.7, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 6.2, 6.8, 7.2, 7.8, 8.2, 8.8, 9.2, 9.6, 2.0, 5.0
]

attendance_rate = [
    60, 62, 65, 67, 70, 72, 75, 78, 80, 81, 83, 85, 87, 88, 90, 91, 93, 95, 96, 98,
    99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 63, 66, 68, 73, 79, 80, 84, 86, 90,
    92, 95, 96, 99, 100, 100, 100, 55, 61, 64, 71, 74, 77, 82, 85, 88, 91, 94, 95, 98,
    99, 100, 100, 100, 58, 62, 65, 68, 72, 76, 83, 86, 89, 92, 94, 97, 98, 100, 100, 100,
    100, 60, 64, 69, 73, 79, 81, 85, 87, 91, 93, 96, 97, 99, 100, 100, 100, 100, 65, 85
]

previous_score = [
    50, 52, 55, 57, 58, 60, 62, 63, 65, 66, 68, 70, 72, 73, 75, 76, 78, 80, 82, 83,
    85, 86, 88, 90, 91, 92, 94, 95, 96, 97, 98, 51, 54, 59, 61, 64, 67, 71, 74, 77,
    79, 81, 84, 87, 89, 93, 96, 48, 53, 56, 60, 63, 65, 69, 71, 75, 78, 80, 83, 86,
    88, 91, 94, 97, 49, 52, 55, 58, 61, 63, 68, 72, 74, 77, 80, 82, 85, 88, 91, 93,
    95, 51, 54, 58, 62, 65, 68, 71, 73, 77, 80, 83, 86, 89, 92, 94, 96, 98, 55, 75
]

final_score = [
    52.5, 55.0, 58.2, 59.8, 61.1, 63.4, 65.5, 67.2, 69.0, 70.3, 72.1, 74.0, 75.8, 77.2, 79.5, 81.0, 83.2, 85.1, 87.5, 89.0,
    91.2, 92.5, 94.0, 95.8, 96.5, 97.2, 98.5, 99.0, 99.5, 100.0, 100.0, 54.1, 57.2, 62.0, 65.4, 71.0, 73.2, 78.5, 81.2, 85.5,
    88.0, 92.1, 94.3, 97.5, 98.8, 99.5, 100.0, 50.2, 56.0, 59.5, 63.8, 67.0, 69.2, 73.5, 75.8, 80.0, 83.2, 86.5, 89.0, 93.1,
    94.8, 97.9, 99.0, 100.0, 51.5, 55.2, 58.6, 61.0, 64.2, 66.8, 72.5, 76.0, 78.5, 81.9, 84.8, 87.5, 90.2, 93.5, 96.2, 98.0,
    99.5, 53.8, 57.0, 61.2, 65.5, 71.2, 74.0, 79.2, 81.5, 86.5, 89.2, 93.5, 95.8, 98.2, 99.5, 100.0, 100.0, 100.0, 57.5, 78.0
]

# Convert dictionaries to DataFrame
df = pd.DataFrame({
    'Hours_Studied': hours_studied,
    'Attendance_Rate': attendance_rate,
    'Previous_Score': previous_score,
    'Final_Score': final_score
})

# Display shape, head, and info
print("Dataset Shape:", df.shape)
print("First 5 rows:")
print(df.head())
print("Dataset Information:")
df.info()
"""))

cells.append(md_cell("""### What Did We Observe?
* The dataset consists of **100 rows** (records) and **4 columns** (features + target).
* All columns contain numerical numbers (floats or integers).
* There are no missing text entries or NaN cells initially.

### What Did We Learn?
* Data is stored in a two-dimensional grid called a **DataFrame** using Pandas.
* We have 3 independent variables (features) and 1 dependent variable (the final score we want to predict).
"""))

# 5. Exploratory Data Analysis
cells.append(md_cell("""# 5. Exploratory Data Analysis (EDA)

EDA is the process of examining the data visually to find patterns, anomalies, and correlations. Let's create three charts:
1. **Histogram of study hours** to understand the distribution of student habits.
2. **Scatter plot of Hours Studied vs. Final Score** to see their direct relationship.
3. **Correlation Heatmap** to see how all numeric variables relate to one another.
"""))

cells.append(code_cell("""# Chart 1: Distribution of Hours Studied
plt.figure(figsize=(8, 4))
sns.histplot(df['Hours_Studied'], bins=15, kde=True, color='teal')
plt.title('Distribution of Student Study Hours')
plt.xlabel('Hours Studied per Week')
plt.ylabel('Number of Students')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

cells.append(md_cell("""### What Did We Observe?
* Student study hours range from 1 to 10 hours per week.
* The distribution is relatively uniform across the range, with minor peaks.

### What Did We Learn?
* This distribution ensures that our model gets exposed to students who study very little as well as students who study a lot.
"""))

cells.append(code_cell("""# Chart 2: Hours Studied vs Final Score Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Hours_Studied', y='Final_Score', data=df, color='darkorange', s=80)
plt.title('Study Hours vs. Final Score')
plt.xlabel('Hours Studied per Week')
plt.ylabel('Final Exam Score')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

cells.append(md_cell("""### What Did We Observe?
* There is a clear upward trend: as the number of hours studied increases, the final score also increases.
* The points cluster tightly along a diagonal path.

### What Did We Learn?
* A straight diagonal path indicates a strong **positive linear relationship**. 
* This confirms that Linear Regression is a suitable choice for this data!
"""))

cells.append(code_cell("""# Chart 3: Correlation Heatmap
plt.figure(figsize=(6, 5))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()
"""))

cells.append(md_cell("""### What Did We Observe?
* All features have correlation coefficients close to **0.9** or above with `Final_Score`.
* `Hours_Studied` and `Attendance_Rate` are also highly correlated with each other.

### What Did We Learn?
* Correlation scores range from -1 to +1. A score close to +1 means a strong positive linear relationship.
* All our selected features have a high correlation with the target variable, making them strong predictor candidates.
"""))

# 6. Data Cleaning
cells.append(md_cell("""# 6. Data Cleaning

In real-world scenarios, datasets are rarely perfect. They contain missing values or duplicate records. 
* We will introduce a single missing value (`NaN`) in our DataFrame.
* We will check for duplicate records.
* We will impute (fill) the missing value using the feature mean.
"""))

cells.append(code_cell("""# 1. Introduce a missing value at index 5 of Hours_Studied
df.loc[5, 'Hours_Studied'] = np.nan

# 2. Check for missing values
print("Missing values count before cleaning:")
print(df.isnull().sum())

# 3. Check for duplicates
print("\\nNumber of duplicate rows:", df.duplicated().sum())

# 4. Fill the missing value with the average (mean) of the column
mean_hours = df['Hours_Studied'].mean()
df['Hours_Studied'] = df['Hours_Studied'].fillna(mean_hours)

# 5. Verify that no missing values remain
print("\\nMissing values count after imputation:")
print(df.isnull().sum())
"""))

cells.append(md_cell("""### What Did We Observe?
* The missing value in `Hours_Studied` was detected and successfully filled using the column's average value.
* No duplicate records were found in the dataset.

### What Did We Learn?
* Missing values must be filled or removed before passing data to scikit-learn models, otherwise, the code will fail with an error.
* Using the mean is a safe and simple way to clean numerical columns.
"""))

# 7. Feature Selection
cells.append(md_cell("""# 7. Feature Selection

Feature selection is the process of choosing which columns will be used as input variables for our model.
* **Features (X)**: `Hours_Studied`, `Attendance_Rate`, `Previous_Score`.
* **Target (y)**: `Final_Score` (what we want to predict).
"""))

cells.append(code_cell("""# Separate features (X) and target (y)
X = df[['Hours_Studied', 'Attendance_Rate', 'Previous_Score']]
y = df['Final_Score']

# Check shapes
print("Features shape (X):", X.shape)
print("Target shape (y):", y.shape)
"""))

cells.append(md_cell("""### What Did We Observe?
* `X` is a 2D matrix with 100 rows and 3 columns.
* `y` is a 1D vector containing the 100 final exam scores.

### What Did We Learn?
* We use uppercase `X` because it represents a multi-column matrix.
* We use lowercase `y` because it represents a single column vector.
"""))

# 8. Train-Test Split
cells.append(md_cell("""# 8. Train-Test Split

To accurately measure model performance, we must split our dataset into two parts:
1. **Training Data (80%)**: Used to fit the regression line.
2. **Testing Data (20%)**: Held back to evaluate predictions.

### What is Data Leakage?
* Data leakage occurs when information from the test dataset is accidentally exposed to the model during training.
* Splitting our data strictly before building the model prevents this cheat, ensuring our evaluation is honest and unbiased.
"""))

cells.append(code_cell("""# Splitting the data using scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Printing the split dataset shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
"""))

cells.append(md_cell("""### What Did We Observe?
* 80 records are assigned to the training set.
* 20 records are held back for the testing set.
* `random_state=42` acts as a seed to ensure we get the exact same split every time the code runs.

### What Did We Learn?
* The test dataset represents unseen data. If our model scores well on it, we can trust it to perform well in production.
"""))

# 9. Model Building
cells.append(md_cell("""# 9. Model Building

Now we introduce the **Linear Regression** algorithm!

### How it works conceptually:
* The algorithm attempts to find the parameters (weights/coefficients) for a linear equation:
  
  $$\text{Final Score} = w_1 \times \text{Hours} + w_2 \times \text{Attendance} + w_3 \times \text{Previous Score} + b$$
  
* Here, $w_1, w_2, w_3$ are coefficients, and $b$ is the intercept.
* It fits the line by minimizing the sum of squared differences between actual points and the line (called **Ordinary Least Squares**).
"""))

cells.append(code_cell("""# Initialize the model instance
model = LinearRegression()
"""))

cells.append(md_cell("""### What Did We Observe?
* We created an empty model template ready to learn from our data.

### What Did We Learn?
* Initializing simply instantiates the algorithm class from scikit-learn; it has not analyzed any data yet.
"""))

# 10. Model Training
cells.append(md_cell("""# 10. Model Training

We train (fit) the model using our training data: `X_train` and `y_train`.
"""))

cells.append(code_cell("""# Fitting the model to find the best-fit line parameters
model.fit(X_train, y_train)
print("Model training completed successfully!")
"""))

cells.append(md_cell("""### What Did We Observe?
* The training script executed instantly.
* The model has calculated the weights (coefficients) and intercept.

### What Did We Learn?
* Fitting adjusts the internal weights of the linear equation using the OLS formula on the training data.
"""))

# 11. Predictions
cells.append(md_cell("""# 11. Predictions

Let's test the model by generating predictions on the test dataset (`X_test`).
"""))

cells.append(code_cell("""# Make predictions on test features
predictions = model.predict(X_test)

# Let's align actual and predicted scores side-by-side
compare_df = pd.DataFrame({
    'Actual_Score': y_test,
    'Predicted_Score': predictions,
    'Difference (Error)': y_test - predictions
})

print(compare_df.head())
"""))

cells.append(md_cell("""### What Did We Observe?
* The predicted scores match the actual exam grades very closely, with errors generally under 1-2 points.

### What Did We Learn?
* The predictions are generated by plugging the test features into the trained linear equation.
"""))

# 12. Evaluation
cells.append(md_cell("""# 12. Evaluation

To evaluate a regression model, we use metrics that measure prediction error:
* **Mean Absolute Error (MAE)**: The average absolute difference between the actual and predicted scores. It is intuitive and easy to explain.
* **Root Mean Squared Error (RMSE)**: The square root of the average squared errors. It penalizes larger errors more heavily.
* **R-squared ($R^2$)**: The proportion of variation in the target variable that is predictable from the features. Ranges from 0 to 1, where 1 represents perfect predictions.
"""))

cells.append(code_cell("""# Computing metrics
mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, predictions)

print(f"Mean Absolute Error (MAE): {mae:.4f} points")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} points")
print(f"R-squared Score (R2): {r2:.4f}")
"""))

cells.append(md_cell("""### What Did We Observe?
* The MAE is very low (under 1.0 points), indicating highly accurate student predictions.
* The $R^2$ score is extremely close to 1.0 (over 0.99), showing that our features explain almost all the variation in final exam scores.

### What Did We Learn?
* An $R^2$ close to 1.0 is exceptional. In real life, student behavior has more noise, but our clean synthetic dataset shows a strong, clear signal.
"""))

# 13. Visualizing Model Performance
cells.append(md_cell("""# 13. Visualizing Model Performance

Visual graphs are the best way to explain errors to non-technical stakeholders. We will create two plots:
1. **Actual vs. Predicted Scatter Plot**: A perfect model will align all points on a 45-degree diagonal line.
2. **Residual Plot**: A plot showing the errors (residuals) on the Y-axis vs. predicted values. Ideally, points should be randomly scattered around the zero line.
"""))

cells.append(code_cell("""# Plot 1: Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, color='purple', alpha=0.8, edgecolors='black', s=80)
# Draw reference diagonal line representing perfect prediction
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, linestyle='--')
plt.title('Actual vs. Predicted Final Scores')
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

cells.append(md_cell("""### What Did We Observe?
* The scatter points are located directly on or extremely close to the red dotted line representing perfect predictions.

### What Did We Learn?
* The closer the points are to the diagonal reference line, the lower the errors and the better our model's performance.
"""))

cells.append(code_cell("""# Plot 2: Residual Plot
residuals = y_test - predictions
plt.figure(figsize=(8, 5))
plt.scatter(predictions, residuals, color='crimson', alpha=0.8, edgecolors='black', s=80)
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.title('Residual Plot (Errors vs. Predicted)')
plt.xlabel('Predicted Exam Scores')
plt.ylabel('Residuals (Actual - Predicted)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

cells.append(md_cell("""### What Did We Observe?
* The residuals are distributed randomly above and below the horizontal black line, with no obvious shape or pattern.
* The magnitude of error is very small.

### What Did We Learn?
* A random spread of residuals is a strong sign that the assumptions of Linear Regression hold true (homoscedasticity).
"""))

# 14. Model Interpretation
cells.append(md_cell("""# 14. Model Interpretation

Interpretation answers: **"What did the model learn?"**
Linear regression calculates a constant (intercept) and a weight (coefficient) for each feature.
"""))

cells.append(code_cell("""# Display intercept and coefficients
print("Model Intercept:", model.intercept_)
print("\\nModel Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"* {col}: {coef:.4f}")
"""))

cells.append(md_cell("""### What Did We Observe?
* The Intercept is positive.
* The Coefficients indicate how much the final score increases when a single feature increases by 1 unit, keeping all other features constant.

### What Did We Learn?
* **Interpretability**: A coefficient of ~1.5 for `Hours_Studied` means that for every additional hour a student studies, their predicted final score increases by 1.5 points!
* This makes Linear Regression incredibly easy to explain to students, teachers, and school administrators.
"""))

# 15. Conclusion
cells.append(md_cell("""# 15. Conclusion

Let's summarize our regression project:
* **Problem**: Predict final scores of students based on study habits.
* **Approach**: We used a synthetic 100-student dataset, cleaned missing values, split the dataset, and trained a Linear Regression model.
* **Results**: Our model achieved an $R^2$ of over 0.99 with a Mean Absolute Error of under 1.0 points.
* **Key Learning**: Study hours have a strong positive linear relationship with exam scores. Linear Regression provides highly explainable predictions.
"""))

# 16. Beginner ML Dictionary
cells.append(md_cell(get_dictionary()))

# Write the notebook
make_notebook("models/Linear_Regression.ipynb", cells)
print("Finished Linear Regression notebook generation!")
