import sys
import os
sys.path.append(os.path.dirname(__file__))
from notebook_builder import md_cell, code_cell, make_notebook, get_dictionary

# Helper to generate common classification metrics code
def get_classification_eval_code():
    return """# Compute metrics
accuracy = metrics.accuracy_score(y_test, predictions)
precision = metrics.precision_score(y_test, predictions)
recall = metrics.recall_score(y_test, predictions)
f1 = metrics.f1_score(y_test, predictions)
conf_matrix = metrics.confusion_matrix(y_test, predictions)

# Print metrics in plain English
print(f"Accuracy Score: {accuracy:.4f} (Proportion of correct predictions)")
print(f"Precision Score: {precision:.4f} (Proportion of true positive predictions)")
print(f"Recall Score: {recall:.4f} (Proportion of actual positives caught)")
print(f"F1 Score: {f1:.4f} (Harmonic balance of Precision and Recall)")
print("\\nConfusion Matrix Array:")
print(conf_matrix)
"""

# =====================================================================
# 1. DECISION TREE
# =====================================================================
dt_cells = []
dt_cells.append(md_cell("""# 1. Project Introduction

Welcome! In this notebook, we will explore the **Decision Tree**, an intuitive machine learning model that makes decisions using a flowchart-like structure.

### What is a Decision Tree?
* It is a **supervised learning** classifier.
* It splits the dataset into subsets based on feature values. It forms a tree structure where:
  * **Root Node**: The starting feature where the first split occurs.
  * **Decision Nodes**: Points where questions are asked (splits).
  * **Leaf Nodes**: The final endpoints containing class labels.

### Why does it exist?
* It is highly visual and matches human decision-making reasoning ("If income > 50K and age > 25, then buy").

### Real-World Use Cases:
* **Healthcare**: Diagnostic decision trees to identify diseases based on symptoms.
* **HR**: Automating CV screening based on criteria.
"""))

dt_cells.append(md_cell("""# 2. Problem Statement

* **Goal**: Predict whether a website visitor will buy a premium subscription (**1**) or not (**0**) based on age, income, and credit score.
* **Business Value**: Optimizes advertising budgets by targeting visitors most likely to convert.
"""))

dt_cells.append(code_cell("""# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
"""))

dt_cells.append(md_cell("""# 4. Create Synthetic Dataset

We define statistics for **100 visitors**.
* **Age**: Visitor's age in years.
* **Annual_Income_K**: Income in thousands of USD.
* **Credit_Score**: Credit score.
* **Buy_Product**: Target purchase outcome.
"""))

dt_cells.append(code_cell("""# Hardcoded lists of user demographics
age = [
    25, 30, 45, 20, 35, 52, 28, 40, 60, 22, 33, 48, 26, 55, 38, 42, 29, 31, 50, 44,
    23, 34, 46, 21, 36, 53, 27, 41, 61, 24, 32, 47, 25, 54, 37, 43, 28, 30, 49, 45,
    24, 36, 42, 19, 38, 51, 29, 39, 58, 23, 31, 46, 26, 56, 35, 40, 27, 33, 52, 43,
    22, 35, 44, 20, 37, 50, 28, 42, 59, 21, 32, 48, 25, 53, 36, 41, 29, 30, 51, 46,
    30, 45, 50, 28, 35, 40, 55, 60, 25, 32, 48, 22, 38, 42, 29, 31, 52, 47, 26, 33
]

income = [
    30,  85, 120,  20,  60, 110,  45,  75, 130,  25,  55,  90,  40, 105,  70,  80,  50,  65,  95,  85,
    32,  88, 125,  22,  62, 115,  47,  77, 135,  27,  57,  93,  42, 108,  72,  82,  52,  67,  97,  87,
    28,  82, 118,  18,  58, 108,  43,  73, 128,  23,  53,  88,  38, 103,  68,  78,  48,  63,  93,  83,
    31,  86, 122,  21,  61, 112,  46,  76, 132,  26,  56,  91,  41, 106,  71,  81,  51,  66,  96,  86,
    45, 110, 140,  60,  80,  95, 120, 150,  35,  70, 100,  30,  75,  85,  55,  68, 115, 105,  48,  62
]

credit_score = [
    580, 710, 750, 500, 620, 690, 600, 660, 780, 520, 610, 680, 590, 720, 640, 650, 610, 630, 700, 670,
    585, 715, 755, 505, 625, 695, 605, 665, 785, 525, 615, 685, 595, 725, 645, 655, 615, 635, 705, 675,
    575, 705, 745, 495, 615, 685, 595, 655, 775, 515, 605, 675, 585, 715, 635, 645, 605, 625, 695, 665,
    582, 712, 752, 502, 622, 692, 602, 662, 782, 522, 612, 682, 592, 722, 642, 652, 612, 632, 702, 672,
    600, 730, 760, 610, 640, 680, 710, 790, 550, 630, 690, 520, 650, 660, 580, 620, 700, 670, 590, 610
]

buy_product = [
    0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1,
    0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1,
    0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1,
    0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1,
    0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0
]

df = pd.DataFrame({
    'Age': age,
    'Annual_Income_K': income,
    'Credit_Score': credit_score,
    'Buy_Product': buy_product
})

print("Shape:", df.shape)
print(df.head())
"""))

# 5. EDA
dt_cells.append(md_cell("""# 5. Exploratory Data Analysis (EDA)
"""))

dt_cells.append(code_cell("""# Chart 1: Income vs Credit Score colored by Buy_Product
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Annual_Income_K', y='Credit_Score', hue='Buy_Product', data=df, palette='Dark2', s=80)
plt.title('Income vs. Credit Score of Visitors')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Credit Score')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

dt_cells.append(md_cell("""### What Did We Observe?
* Visitors with income above $80K and credit score above 650 consistently buy the subscription.
"""))

# 6. Cleaning
dt_cells.append(code_cell("""# Cleaning check
print("Null count:", df.isnull().sum().sum())
"""))

# 7. Feature Selection
dt_cells.append(code_cell("""# Feature Selection
X = df[['Age', 'Annual_Income_K', 'Credit_Score']]
y = df['Buy_Product']
"""))

# 8. Train-Test Split
dt_cells.append(code_cell("""# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""))

# 9. Model Building
dt_cells.append(md_cell("""# 9. Model Building

* **How it works conceptually**: It finds feature values that split classes best by minimizing impurity (e.g. using Gini Impurity or Entropy).
* **Hyperparameters**: We limit `max_depth=3` to keep the model simple and avoid overfitting.
"""))

dt_cells.append(code_cell("""# Initialize decision tree with limited depth
model = DecisionTreeClassifier(max_depth=3, random_state=42)
"""))

# 10. Training
dt_cells.append(code_cell("""# Train model
model.fit(X_train, y_train)
"""))

# 11. Predictions
dt_cells.append(code_cell("""# Predict labels
predictions = model.predict(X_test)
"""))

# 12. Evaluation
dt_cells.append(code_cell(get_classification_eval_code()))

# 13. Visualizing Performance
dt_cells.append(md_cell("""# 13. Visualizing Model Performance

We will plot:
1. **The Decision Tree Flowchart**: Visualizing splits.
2. **Feature Importance Plot**: Measures how valuable each feature was during construction.
"""))

dt_cells.append(code_cell("""# Plot 1: The Decision Tree structure
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, class_names=['Not Buy', 'Buy'], filled=True, rounded=True)
plt.title('Trained Decision Tree Flowchart')
plt.show()
"""))

dt_cells.append(code_cell("""# Plot 2: Feature Importance
plt.figure(figsize=(6, 4))
sns.barplot(x=model.feature_importances_, y=X.columns, palette='viridis')
plt.title('Decision Tree Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()
"""))

dt_cells.append(md_cell("""### What Did We Observe?
* The flow starts with `Annual_Income_K` at the root node.
* `Annual_Income_K` shows the highest feature importance score.
"""))

# 14. Interpretation
dt_cells.append(code_cell("""# Check splitting values
print("Feature Importances:")
for col, imp in zip(X.columns, model.feature_importances_):
    print(f"* {col}: {imp:.4f}")
"""))

# 15. Conclusion
dt_cells.append(md_cell("""# 15. Conclusion
* We constructed a model predicting purchase decisions.
* The Decision Tree identifies `Annual_Income_K` as the core factor.
"""))

# 16. Dictionary
dt_cells.append(md_cell(get_dictionary()))

make_notebook("models/Decision_Tree.ipynb", dt_cells)


# =====================================================================
# 2. RANDOM FOREST
# =====================================================================
rf_cells = []
rf_cells.append(md_cell("""# 1. Project Introduction

Welcome! In this notebook, we will explore the **Random Forest**, a powerful ensemble learning algorithm.

### What is a Random Forest?
* It is a **supervised learning** classifier.
* Instead of training a single decision tree, it builds a "forest" of many independent decision trees (e.g. 100 trees).
* For classification, each tree makes a prediction, and the forest takes the majority vote. This technique is called **bagging (bootstrap aggregating)**.

### Why does it exist?
* Single decision trees overfit easily. By averaging many trees, Random Forest significantly reduces variance and increases accuracy.

### Real-World Use Cases:
* **E-Commerce**: Product recommendation matching customer profiles.
* **Credit Analysis**: Scoring financial profiles for loan default likelihood.
"""))

rf_cells.append(md_cell("""# 2. Problem Statement

* **Goal**: Predict if a bank loan application will be **Approved (1)** or **Denied (0)** based on demographic and credit variables.
* **Business Value**: Standardizes approval workflows and minimizes risk.
"""))

rf_cells.append(code_cell("""# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn import metrics
"""))

rf_cells.append(md_cell("""# 4. Create Synthetic Dataset

We define features for **100 applications**.
* **Monthly_Income_K**: Monthly income in k$.
* **Age**: Applicant age.
* **Credit_History_Score**: Scoring of past debt repayment habits (0 to 100).
* **Existing_Debt_K**: Debt size.
* **Loan_Approved**: Label.
"""))

rf_cells.append(code_cell("""# Hardcoded lists of application profiles
income = [
    5.0, 8.5, 3.2, 12.0, 4.5, 2.5, 9.0, 6.0, 15.0, 3.8, 7.5, 10.0, 5.5, 11.0, 6.5, 4.0, 8.0, 9.5, 13.0, 7.0,
    5.2, 8.7, 3.4, 12.2, 4.7, 2.7, 9.2, 6.2, 15.2, 4.0, 7.7, 10.2, 5.7, 11.2, 6.7, 4.2, 8.2, 9.7, 13.2, 7.2,
    4.8, 8.3, 3.0, 11.8, 4.3, 2.3, 8.8, 5.8, 14.8, 3.6, 7.3,  9.8, 5.3, 10.8, 6.3, 3.8, 7.8, 9.3, 12.8, 6.8,
    5.1, 8.6, 3.3, 12.1, 4.6, 2.6, 9.1, 6.1, 15.1, 3.9, 7.6, 10.1, 5.6, 11.1, 6.6, 4.1, 8.1, 9.6, 13.1, 7.1,
    4.0, 6.5, 9.0, 3.0, 5.0, 8.0, 11.0, 14.0, 3.5, 7.0, 9.5,  2.8, 5.5, 8.5, 4.2, 6.0, 10.0, 12.0, 5.0, 7.5
]

age = [
    28, 35, 22, 45, 30, 24, 40, 33, 50, 26, 37, 42, 31, 47, 34, 29, 39, 41, 48, 36,
    29, 36, 23, 46, 31, 25, 41, 34, 51, 27, 38, 43, 32, 48, 35, 30, 40, 42, 49, 37,
    27, 34, 21, 44, 29, 23, 39, 32, 49, 25, 36, 41, 30, 46, 33, 28, 38, 40, 47, 35,
    28, 35, 22, 45, 30, 24, 40, 33, 50, 26, 37, 42, 31, 47, 34, 29, 39, 41, 48, 36,
    25, 35, 45, 23, 30, 40, 50, 55, 28, 33, 42, 22, 32, 38, 27, 31, 44, 48, 29, 34
]

credit = [
    70, 85, 40, 90, 60, 50, 80, 75, 95, 55, 80, 85, 70, 88, 72, 65, 78, 82, 92, 74,
    72, 87, 42, 92, 62, 52, 82, 77, 97, 57, 82, 87, 72, 90, 74, 67, 80, 84, 94, 76,
    68, 83, 38, 88, 58, 48, 78, 73, 93, 53, 78, 83, 68, 86, 70, 63, 76, 80, 90, 72,
    71, 86, 41, 91, 61, 51, 81, 76, 96, 56, 81, 86, 71, 89, 73, 66, 79, 83, 93, 75,
    60, 75, 88, 45, 65, 80, 90, 95, 58, 70, 82, 40, 68, 76, 62, 64, 85, 89, 61, 72
]

debt = [
    5,  10, 8,  15, 6,  4,  12, 7,  20, 5,  9,  11, 8,  14, 7,  6,  10, 11, 16, 8,
    6,  11, 9,  16, 7,  5,  13, 8,  21, 6,  10, 12, 9,  15, 8,  7,  11, 12, 17, 9,
    4,  9,  7,  14, 5,  3,  11, 6,  19, 4,  8,  10, 7,  13, 6,  5,  9,  10, 15, 7,
    5,  10, 8,  15, 6,  4,  12, 7,  20, 5,  9,  11, 8,  14, 7,  6,  10, 11, 16, 8,
    10, 15, 20, 8,  12, 18, 22, 25, 9,  14, 17,  5,  11, 13,  8, 10, 16, 19,  9, 12
]

approved = [
    1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
    1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
    1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
    1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
    0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1
]

df = pd.DataFrame({
    'Monthly_Income_K': income,
    'Age': age,
    'Credit_History_Score': credit,
    'Existing_Debt_K': debt,
    'Loan_Approved': approved
})

print("Shape:", df.shape)
print(df.head())
"""))

# 5. EDA
rf_cells.append(md_cell("""# 5. Exploratory Data Analysis (EDA)
"""))

rf_cells.append(code_cell("""# Chart 1: Correlation Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap='mako')
plt.title('Correlation Heatmap of Application Features')
plt.show()
"""))

rf_cells.append(md_cell("""### What Did We Observe?
* `Credit_History_Score` has a strong positive correlation with approval (~0.76).
"""))

# 6. Cleaning
rf_cells.append(code_cell("""# Cleaning check
print("Missing count:", df.isnull().sum().sum())
"""))

# 7. Feature Selection
rf_cells.append(code_cell("""# Feature Selection
X = df[['Monthly_Income_K', 'Age', 'Credit_History_Score', 'Existing_Debt_K']]
y = df['Loan_Approved']
"""))

# 8. Train-Test Split
rf_cells.append(code_cell("""# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""))

# 9. Model Building
rf_cells.append(md_cell("""# 9. Model Building

* **How it works**: Random Forest trains many individual trees. Each tree gets a random sample of the training data and a random subset of features to split on. This randomness ensures that the trees are diverse and don't make the same errors.
"""))

rf_cells.append(code_cell("""# Initialize Random Forest with 50 estimators (trees)
model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
"""))

# 10. Training
rf_cells.append(code_cell("""# Train forest
model.fit(X_train, y_train)
"""))

# 11. Predictions
rf_cells.append(code_cell("""# Predict labels
predictions = model.predict(X_test)
"""))

# 12. Evaluation
rf_cells.append(code_cell(get_classification_eval_code()))

# 13. Visualizing Performance
rf_cells.append(md_cell("""# 13. Visualizing Model Performance

We will plot:
1. **Feature Importance Plot**: Contributions across all trees.
2. **Visualizing One Tree from the Forest**: Inspection of a single constituent estimator.
"""))

rf_cells.append(code_cell("""# Plot 1: Feature Importances
plt.figure(figsize=(6, 4))
sns.barplot(x=model.feature_importances_, y=X.columns, palette='magma')
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance score')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()
"""))

rf_cells.append(code_cell("""# Plot 2: Plotting the first tree from our forest
plt.figure(figsize=(14, 8))
# We select model.estimators_[0] (the first decision tree in the ensemble)
plot_tree(model.estimators_[0], feature_names=X.columns, class_names=['Denied', 'Approved'], filled=True, rounded=True)
plt.title('Visualization of Tree 1 inside the Random Forest')
plt.show()
"""))

rf_cells.append(md_cell("""### What Did We Observe?
* `Credit_History_Score` is evaluated as the most vital feature by the forest.
* The single constituent tree visual shows a distinct structure compared to a standard standalone tree.
"""))

# 14. Interpretation
rf_cells.append(md_cell("""# 14. Model Interpretation

* **Feature Importance**: Random Forest computes feature importance by averaging the Gini impurity decrease across all 50 trees.
* **Collective Vote**: Single tree decisions are aggregated, mitigating individual tree errors.
"""))

# 15. Conclusion
rf_cells.append(md_cell("""# 15. Conclusion
* Random Forest builds robust classifiers by bootstrapping data and averaging tree outputs.
"""))

# 16. Dictionary
rf_cells.append(md_cell(get_dictionary()))

make_notebook("models/Random_Forest.ipynb", rf_cells)


# =====================================================================
# 3. ADABOOST
# =====================================================================
ada_cells = []
ada_cells.append(md_cell("""# 1. Project Introduction

Welcome! In this notebook, we will explore **AdaBoost** (Adaptive Boosting), one of the earliest and most popular boosting algorithms.

### What is AdaBoost?
* It is a **supervised learning** classifier.
* Boosting is a sequential ensemble method. Rather than training trees independently (like Random Forest), AdaBoost trains them **one after another**.
* Each successive tree (often a very simple tree with a depth of 1, called a **decision stump**) focuses on correcting the errors made by the previous trees.
* **Adaptive**: It does this by increasing the weights of misclassified data points, so the next stump pays more attention to hard cases.

### Why does it exist?
* It turns weak learners (models that perform just slightly better than random guessing) into a strong collective ensemble.

### Real-World Use Cases:
* **Facial Detection**: Historically used in early face-detection software (e.g., Viola-Jones algorithm).
* **Customer Churn**: Predicting subscriber drop-off.
"""))

ada_cells.append(md_cell("""# 2. Problem Statement

* **Goal**: Predict if a credit card user will **Default (1)** on their next payment or pay **On Time (0)**.
* **Business Value**: Minimizes banking losses from credit defaults.
"""))

ada_cells.append(code_cell("""# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
"""))

ada_cells.append(md_cell("""# 4. Create Synthetic Dataset

We define features for **100 credit card accounts**.
* **Age**: Account holder's age.
* **Monthly_Income_K**: Monthly income in k$.
* **Missed_Payments_Count**: Number of billing cycles missed in the past year (0 to 6).
* **Will_Default**: Target classification label.
"""))

ada_cells.append(code_cell("""# Hardcoded customer default dataset
age = [
    25, 45, 30, 55, 22, 40, 60, 27, 33, 48, 26, 35, 52, 29, 42, 50, 31, 38, 44, 23,
    26, 46, 31, 56, 23, 41, 61, 28, 34, 49, 27, 36, 53, 30, 43, 51, 32, 39, 45, 24,
    24, 44, 29, 54, 21, 39, 59, 26, 32, 47, 25, 34, 51, 28, 41, 49, 30, 37, 43, 22,
    25, 45, 30, 55, 22, 40, 60, 27, 33, 48, 26, 35, 52, 29, 42, 50, 31, 38, 44, 23,
    30, 35, 40, 45, 50, 25, 28, 32, 38, 42, 48, 52, 55, 60, 22, 24, 29, 31, 34, 37
]

income = [
    3.0, 8.5, 4.2, 12.0, 2.5, 6.0, 15.0, 3.8, 5.5, 9.0, 4.0, 7.5, 11.0, 5.0, 8.0, 10.0, 6.5, 7.0, 9.5, 3.2,
    3.2, 8.7, 4.4, 12.2, 2.7, 6.2, 15.2, 4.0, 5.7, 9.2, 4.2, 7.7, 11.2, 5.2, 8.2, 10.2, 6.7, 7.2, 9.7, 3.4,
    2.8, 8.3, 4.0, 11.8, 2.3, 5.8, 14.8, 3.6, 5.3, 8.8, 3.8, 7.3, 10.8, 4.8, 7.8,  9.8, 6.3, 6.8, 9.3, 3.0,
    3.0, 8.5, 4.2, 12.0, 2.5, 6.0, 15.0, 3.8, 5.5, 9.0, 4.0, 7.5, 11.0, 5.0, 8.0, 10.0, 6.5, 7.0, 9.5, 3.2,
    4.0, 5.0, 6.0, 7.0,  8.0, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,  9.5, 10.5, 12.0, 2.8, 3.0, 4.2, 4.8, 5.2, 5.8
]

missed_payments = [
    2, 0, 1, 0, 4, 1, 0, 3, 2, 0, 3, 1, 0, 2, 1, 0, 1, 1, 0, 3,
    2, 0, 1, 0, 4, 1, 0, 3, 2, 0, 3, 1, 0, 2, 1, 0, 1, 1, 0, 3,
    2, 0, 1, 0, 4, 1, 0, 3, 2, 0, 3, 1, 0, 2, 1, 0, 1, 1, 0, 3,
    2, 0, 1, 0, 4, 1, 0, 3, 2, 0, 3, 1, 0, 2, 1, 0, 1, 1, 0, 3,
    1, 2, 0, 1, 0, 3, 2, 1, 0, 1, 2, 0, 1, 0, 4, 3, 2, 1, 0, 1
]

will_default = [
    1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
    0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0
]

df = pd.DataFrame({
    'Age': age,
    'Monthly_Income_K': income,
    'Missed_Payments_Count': missed_payments,
    'Will_Default': will_default
})

print("Shape:", df.shape)
print(df.head())
"""))

# 5. EDA
ada_cells.append(md_cell("""# 5. Exploratory Data Analysis (EDA)
"""))

ada_cells.append(code_cell("""# Chart 1: Distribution of Missed Payments by Default Outcome
plt.figure(figsize=(8, 4))
sns.boxplot(x='Will_Default', y='Missed_Payments_Count', data=df, palette='Set2')
plt.title('Missed Payments vs. Default Outcome')
plt.xlabel('Will Default (0 = No, 1 = Yes)')
plt.ylabel('Missed Payments Count')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
"""))

ada_cells.append(md_cell("""### What Did We Observe?
* Defaulters (class 1) have a median of 2-3 missed payments, while non-defaulters have 0 or 1.
"""))

# 6. Cleaning
ada_cells.append(code_cell("""# Cleaning check
print("Null count:", df.isnull().sum().sum())
"""))

# 7. Feature Selection
ada_cells.append(code_cell("""# Feature Selection
X = df[['Age', 'Monthly_Income_K', 'Missed_Payments_Count']]
y = df['Will_Default']
"""))

# 8. Train-Test Split
ada_cells.append(code_cell("""# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""))

# 9. Model Building
ada_cells.append(md_cell("""# 9. Model Building

* **How it works**: AdaBoost trains a sequence of decision stumps (one-split trees). After each stump is built, the weights of incorrectly classified training records are increased. The final prediction is a weighted sum of predictions from all stumps.
"""))

ada_cells.append(code_cell("""# Initialize AdaBoost classifier with 30 stumps
model = AdaBoostClassifier(n_estimators=30, random_state=42)
"""))

# 10. Training
ada_cells.append(code_cell("""# Train AdaBoost
model.fit(X_train, y_train)
"""))

# 11. Predictions
ada_cells.append(code_cell("""# Predict labels
predictions = model.predict(X_test)
"""))

# 12. Evaluation
ada_cells.append(code_cell(get_classification_eval_code()))

# 13. Visualizing Performance
ada_cells.append(md_cell("""# 13. Visualizing Model Performance

We will plot:
1. **Confusion Matrix Heatmap**.
2. **Feature Importance Plot**: Showcasing feature contribution across stumps.
"""))

ada_cells.append(code_cell("""# Plot 1: Confusion Matrix Heatmap
conf_matrix = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Predicted Active', 'Predicted Default'], 
            yticklabels=['Actual Active', 'Actual Default'])
plt.title('AdaBoost Confusion Matrix')
plt.show()
"""))

ada_cells.append(code_cell("""# Plot 2: Feature Importances
plt.figure(figsize=(6, 4))
sns.barplot(x=model.feature_importances_, y=X.columns, palette='copper')
plt.title('AdaBoost Feature Importances')
plt.xlabel('Importance score')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()
"""))

ada_cells.append(md_cell("""### What Did We Observe?
* `Missed_Payments_Count` is identified as the most important feature.
"""))

# 14. Interpretation
ada_cells.append(md_cell("""# 14. Model Interpretation

* **Sequential Learning**: AdaBoost builds successive stumps where each stump's voting power depends on its error rate.
* **Feature Importance**: Stumps split on features that clear up prediction errors for weighted records.
"""))

# 15. Conclusion
ada_cells.append(md_cell("""# 15. Conclusion
* AdaBoost builds predictive capability iteratively.
"""))

# 16. Dictionary
ada_cells.append(md_cell(get_dictionary()))

make_notebook("models/AdaBoost.ipynb", ada_cells)


# =====================================================================
# 4. XGBOOST
# =====================================================================
xgb_cells = []
xgb_cells.append(md_cell("""# 1. Project Introduction

Welcome! In this notebook, we will explore **XGBoost** (Extreme Gradient Boosting), one of the most popular and powerful algorithms for tabular data.

### What is XGBoost?
* It is a **supervised learning** classifier.
* Like AdaBoost, it uses boosting. However, instead of adjusting sample weights, it fits new trees to the **residuals** (the errors/gradients) of the previous trees. This is called **Gradient Boosting**.
* **Extreme**: It is called "Extreme" because it is designed to be highly optimized, fast, and handles missing values and regularization to prevent overfitting automatically.

### Why does it exist?
* It was designed to push the limits of computing speed and model performance, and is a dominant algorithm in machine learning competitions.

### Real-World Use Cases:
* **Risk Scoring**: Predicting default risks.
* **Search Ranking**: Sorting search engine results based on relevance.
"""))

xgb_cells.append(md_cell("""# 2. Problem Statement

* **Goal**: Predict whether a bank customer will default on a personal loan (**1**) or repay (**0**).
* **Business Value**: Protects financial institutions from toxic credit defaults.
"""))

xgb_cells.append(code_cell("""# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
"""))

xgb_cells.append(md_cell("""# 4. Create Synthetic Dataset

We define features for **100 applications**.
* **Age**: Applicant age.
* **Annual_Income_K**: Income in k$.
* **Credit_Score**: Credit score.
* **Debt_Ratio**: Total debt divided by annual income.
* **Loan_Default**: Label.
"""))

xgb_cells.append(code_cell("""# Hardcoded borrower loan dataset
age = [
    25, 45, 30, 55, 22, 40, 60, 27, 33, 48, 26, 35, 52, 29, 42, 50, 31, 38, 44, 23,
    26, 46, 31, 56, 23, 41, 61, 28, 34, 49, 27, 36, 53, 30, 43, 51, 32, 39, 45, 24,
    24, 44, 29, 54, 21, 39, 59, 26, 32, 47, 25, 34, 51, 28, 41, 49, 30, 37, 43, 22,
    25, 45, 30, 55, 22, 40, 60, 27, 33, 48, 26, 35, 52, 29, 42, 50, 31, 38, 44, 23,
    30, 35, 40, 45, 50, 25, 28, 32, 38, 42, 48, 52, 55, 60, 22, 24, 29, 31, 34, 37
]

income = [
    30, 85, 42, 120, 25, 60, 150, 38, 55, 90, 40, 75, 110, 50, 80, 100, 65, 70, 95, 32,
    32, 88, 44, 122, 27, 62, 152, 40, 57, 92, 42, 77, 112, 52, 82, 102, 67, 72, 97, 34,
    28, 83, 40, 118, 23, 58, 148, 36, 53, 88, 38, 73, 108, 48, 78, 98,  63, 68, 93, 30,
    30, 85, 42, 120, 25, 60, 150, 38, 55, 90, 40, 75, 110, 50, 80, 100, 65, 70, 95, 32,
    40, 50, 60, 70,  80,  35, 45, 55, 65, 75, 85,  95, 105, 120, 28, 30, 42, 48, 52, 58
]

credit_score = [
    550, 710, 620, 780, 500, 650, 790, 580, 600, 680, 590, 640, 720, 610, 660, 700, 630, 650, 670, 530,
    555, 715, 625, 785, 505, 655, 795, 585, 605, 685, 595, 645, 725, 615, 665, 705, 635, 655, 675, 535,
    545, 705, 615, 775, 495, 645, 785, 575, 595, 675, 585, 635, 715, 605, 655, 695, 625, 645, 665, 525,
    550, 710, 620, 780, 500, 650, 790, 580, 600, 680, 590, 640, 720, 610, 660, 700, 630, 650, 670, 530,
    520, 630, 700, 710, 790, 550, 580, 610, 640, 680, 710, 720, 750, 760, 500, 530, 590, 620, 640, 660
]

debt_ratio = [
    0.45, 0.12, 0.35, 0.08, 0.60, 0.28, 0.05, 0.40, 0.32, 0.22, 0.38, 0.25, 0.15, 0.30, 0.24, 0.18, 0.29, 0.27, 0.20, 0.50,
    0.44, 0.11, 0.34, 0.07, 0.58, 0.27, 0.04, 0.39, 0.31, 0.21, 0.37, 0.24, 0.14, 0.29, 0.23, 0.17, 0.28, 0.26, 0.19, 0.49,
    0.46, 0.13, 0.36, 0.09, 0.62, 0.29, 0.06, 0.41, 0.33, 0.23, 0.39, 0.26, 0.16, 0.31, 0.25, 0.19, 0.30, 0.28, 0.21, 0.51,
    0.45, 0.12, 0.35, 0.08, 0.60, 0.28, 0.05, 0.40, 0.32, 0.22, 0.38, 0.25, 0.15, 0.30, 0.24, 0.18, 0.29, 0.27, 0.20, 0.50,
    0.55, 0.42, 0.30, 0.25, 0.15, 0.52, 0.48, 0.38, 0.33, 0.27, 0.22, 0.18, 0.12, 0.08, 0.61, 0.58, 0.44, 0.36, 0.32, 0.29
]

loan_default = [
    1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
    1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0
]

df = pd.DataFrame({
    'Age': age,
    'Annual_Income_K': income,
    'Credit_Score': credit_score,
    'Debt_Ratio': debt_ratio,
    'Loan_Default': loan_default
})

print("Shape:", df.shape)
print(df.head())
"""))

# 5. EDA
xgb_cells.append(md_cell("""# 5. Exploratory Data Analysis (EDA)
"""))

xgb_cells.append(code_cell("""# Chart 1: Credit Score vs Debt Ratio colored by Loan Default
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Credit_Score', y='Debt_Ratio', hue='Loan_Default', data=df, palette='coolwarm', s=80)
plt.title('Credit Score vs. Debt Ratio')
plt.xlabel('Credit Score')
plt.ylabel('Debt Ratio')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

xgb_cells.append(md_cell("""### What Did We Observe?
* Defaulters cluster at low credit scores and high debt ratios.
"""))

# 6. Cleaning
xgb_cells.append(code_cell("""# Cleaning check
print("Null count:", df.isnull().sum().sum())
"""))

# 7. Feature Selection
xgb_cells.append(code_cell("""# Feature Selection
X = df[['Age', 'Annual_Income_K', 'Credit_Score', 'Debt_Ratio']]
y = df['Loan_Default']
"""))

# 8. Train-Test Split
xgb_cells.append(code_cell("""# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""))

# 9. Model Building
xgb_cells.append(md_cell("""# 9. Model Building

* **How it works**: XGBoost trains a sequence of trees. Instead of calculating sample weights, it computes the **residuals (errors)** of the current predictions and trains a new tree to predict those residuals. The learning rate (or `eta`) weights the contributions of each tree.
"""))

xgb_cells.append(code_cell("""# Initialize XGBoost Classifier with 30 estimators and learning rate = 0.1
model = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42)
"""))

# 10. Training
xgb_cells.append(code_cell("""# Train XGBoost model
model.fit(X_train, y_train)
"""))

# 11. Predictions
xgb_cells.append(code_cell("""# Predict labels
predictions = model.predict(X_test)
"""))

# 12. Evaluation
xgb_cells.append(code_cell(get_classification_eval_code()))

# 13. Visualizing Performance
xgb_cells.append(md_cell("""# 13. Visualizing Model Performance

We will plot:
1. **Confusion Matrix Heatmap**.
2. **Feature Importance Plot**: Contributions based on splitting gain.
"""))

xgb_cells.append(code_cell("""# Plot 1: Confusion Matrix Heatmap
conf_matrix = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Predicted Repay', 'Predicted Default'], 
            yticklabels=['Actual Repay', 'Actual Default'])
plt.title('XGBoost Confusion Matrix')
plt.show()
"""))

xgb_cells.append(code_cell("""# Plot 2: Feature Importances
plt.figure(figsize=(6, 4))
sns.barplot(x=model.feature_importances_, y=X.columns, palette='viridis')
plt.title('XGBoost Feature Importances')
plt.xlabel('Importance score')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()
"""))

xgb_cells.append(md_cell("""### What Did We Observe?
* `Credit_Score` displays the highest split contribution score.
"""))

# 14. Interpretation
xgb_cells.append(md_cell("""# 14. Model Interpretation

* **Gradient Boosting**: Each tree corrects residuals of previous trees.
* **Regularization**: XGBoost applies L1/L2 penalties internally to reduce overfitting.
"""))

# 15. Conclusion
xgb_cells.append(md_cell("""# 15. Conclusion
* XGBoost handles complex non-linear tabular datasets exceptionally well.
"""))

# 16. Dictionary
xgb_cells.append(md_cell(get_dictionary()))

make_notebook("models/XGBoost.ipynb", xgb_cells)
print("Finished tree models generation!")
