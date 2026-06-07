import sys
import os
sys.path.append(os.path.dirname(__file__))
from notebook_builder import md_cell, code_cell, make_notebook, get_dictionary

# Helper to generate common classification metrics code
def get_classification_eval_code(model_name):
    return f"""# Compute classification evaluation metrics
accuracy = metrics.accuracy_score(y_test, predictions)
precision = metrics.precision_score(y_test, predictions)
recall = metrics.recall_score(y_test, predictions)
f1 = metrics.f1_score(y_test, predictions)
conf_matrix = metrics.confusion_matrix(y_test, predictions)

# Print metrics in plain English
print(f"Accuracy Score: {{accuracy:.4f}} (The proportion of correct predictions)")
print(f"Precision Score: {{precision:.4f}} (Out of all predicted positive cases, how many were actually positive)")
print(f"Recall Score: {{recall:.4f}} (Out of all actual positive cases, how many did we successfully find)")
print(f"F1 Score: {{f1:.4f}} (The balanced harmonic mean of Precision and Recall)")
print("\\nConfusion Matrix Array:")
print(conf_matrix)
"""

# =====================================================================
# 1. LOGISTIC REGRESSION
# =====================================================================
lr_cells = []
lr_cells.append(md_cell("""# 1. Project Introduction

Welcome! In this notebook, we will explore **Logistic Regression**, the classic baseline algorithm for binary classification problems.

### What is Logistic Regression?
* It is a **supervised learning** classification algorithm.
* Despite the word "Regression" in its name, it is used for **classification** (predicting labels, not numbers).
* It estimates the probability that a data point belongs to a particular class (e.g., probability of passing). If the probability is greater than 50%, it predicts a positive class (1); otherwise, a negative class (0).

### Why does it exist?
* It acts as a bridge between linear models and probability, allowing us to draw decision boundaries between categories.

### Real-World Use Cases:
* **Healthcare**: Predicting if a tumor is malignant (1) or benign (0).
* **Finance**: Classifying transaction logs as fraud (1) or safe (0).
* **Marketing**: Classifying a user as subscriber (1) or non-subscriber (0).
"""))

lr_cells.append(md_cell("""# 2. Problem Statement

* **Goal**: Predict whether a student will **Pass (1)** or **Fail (0)** a course based on study habits.
* **Business Value**: Allows educators to proactively identify students needing academic assistance.
"""))

lr_cells.append(code_cell("""# Importing libraries step-by-step
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
"""))

lr_cells.append(md_cell("""# 4. Create Synthetic Dataset

We create a hardcoded dataset representing study statistics of **100 students**.
* **Hours_Studied**: Hours spent studying per week.
* **Attendance_Rate**: Attendance percentage.
* **Assignments_Submitted**: Number of assignments submitted (out of 10).
* **Pass_Fail**: Pass (1) or Fail (0).
"""))

lr_cells.append(code_cell("""# Hardcoded lists of student data
hours_studied = [
    1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.2, 4.5, 4.8, 5.0, 5.2, 5.5, 5.8, 6.0,
    6.2, 6.5, 6.8, 7.0, 7.2, 7.5, 7.8, 8.0, 8.2, 8.5, 8.8, 9.0, 9.2, 9.5, 1.1, 1.3, 1.7, 2.1, 2.4, 2.7,
    3.1, 3.4, 3.9, 4.1, 4.6, 5.1, 5.6, 6.1, 6.7, 7.1, 7.6, 8.1, 8.6, 9.1, 1.4, 1.9, 2.3, 2.6, 2.9, 3.3,
    3.6, 4.3, 4.7, 5.3, 5.7, 6.3, 6.6, 7.3, 7.7, 8.3, 8.7, 9.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.4, 4.9,
    5.4, 5.9, 6.4, 6.9, 7.4, 7.9, 8.4, 8.9, 9.4, 1.5, 2.6, 3.7, 4.8, 5.9, 7.0, 8.1, 9.2, 2.0, 5.0, 8.0
]

attendance_rate = [
    55, 58, 60, 62, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82, 84, 85, 87, 89,
    90, 91, 93, 95, 96, 98, 99, 100, 100, 100, 100, 100, 100, 100, 56, 59, 61, 64, 66, 69,
    71, 73, 76, 78, 81, 83, 86, 88, 91, 93, 95, 97, 99, 100, 57, 60, 62, 65, 68, 70,
    72, 77, 80, 83, 85, 88, 90, 94, 96, 98, 99, 100, 58, 61, 65, 70, 73, 76, 79, 82,
    85, 87, 89, 92, 94, 96, 98, 99, 100, 60, 68, 75, 82, 89, 95, 99, 100, 62, 80, 95
]

assignments_submitted = [
    2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 2, 3, 3, 4, 4, 5,
    5, 6, 6, 7, 7, 8, 8, 9, 9, 9, 10, 10, 10, 10, 3, 4, 4, 5, 5, 6,
    6, 7, 7, 8, 8, 9, 9, 10, 10, 10, 10, 10, 3, 4, 5, 5, 6, 7, 7, 8,
    8, 9, 9, 10, 10, 10, 10, 10, 10, 3, 5, 6, 8, 9, 10, 10, 10, 4, 7, 9
]

pass_fail = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1
]

df = pd.DataFrame({
    'Hours_Studied': hours_studied,
    'Attendance_Rate': attendance_rate,
    'Assignments_Submitted': assignments_submitted,
    'Pass_Fail': pass_fail
})

print("Dataset Shape:", df.shape)
print("First 5 rows:")
print(df.head())
print("Dataset Info:")
df.info()
"""))

lr_cells.append(md_cell("""### What Did We Observe?
* The dataset has **100 rows** and **4 columns**.
* `Pass_Fail` is binary containing only values `0` (Fail) and `1` (Pass).

### What Did We Learn?
* This classification task maps study features to a discrete binary class target.
"""))

lr_cells.append(md_cell("""# 5. Exploratory Data Analysis (EDA)

We explore the distribution of classes and check the relationship between features.
"""))

lr_cells.append(code_cell("""# Chart 1: Pass/Fail Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Pass_Fail', data=df, palette='Set2')
plt.title('Count of Pass vs. Fail Students')
plt.xlabel('Academic Outcome (0 = Fail, 1 = Pass)')
plt.ylabel('Number of Students')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
"""))

lr_cells.append(md_cell("""### What Did We Observe?
* There are more passing students (~60) than failing students (~40) in the dataset.

### What Did We Learn?
* The classes are relatively balanced, meaning we can use Accuracy as a reliable starting metric.
"""))

lr_cells.append(code_cell("""# Chart 2: Hours Studied vs Attendance Colored by Pass/Fail
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Hours_Studied', y='Attendance_Rate', hue='Pass_Fail', data=df, palette='coolwarm', s=80)
plt.title('Hours Studied vs. Attendance Rate')
plt.xlabel('Hours Studied')
plt.ylabel('Attendance Rate (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

lr_cells.append(md_cell("""### What Did We Observe?
* Failing students (blue points) cluster at the bottom-left corner (low hours, low attendance).
* Passing students (red points) occupy the upper-right area.

### What Did We Learn?
* A line can separate these two clusters. This makes Logistic Regression a great choice.
"""))

# 6. Data Cleaning
lr_cells.append(md_cell("""# 6. Data Cleaning

Let's check for missing values and duplicates.
"""))

lr_cells.append(code_cell("""# Clean duplicates and check missing cells
df.loc[10, 'Attendance_Rate'] = np.nan
print("Missing values count:")
print(df.isnull().sum())

# Impute missing attendance with the median value
df['Attendance_Rate'] = df['Attendance_Rate'].fillna(df['Attendance_Rate'].median())

print("\\nMissing values count after cleaning:")
print(df.isnull().sum())
"""))

lr_cells.append(md_cell("""### What Did We Observe?
* Imputation using the median was successful.

### What Did We Learn?
* Filling missing values ensures that the model building pipeline doesn't break.
"""))

# 7. Feature Selection
lr_cells.append(code_cell("""# Feature Selection
X = df[['Hours_Studied', 'Attendance_Rate', 'Assignments_Submitted']]
y = df['Pass_Fail']
"""))

# 8. Train-Test Split
lr_cells.append(code_cell("""# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""))

# 9. Model Building
lr_cells.append(md_cell("""# 9. Model Building

* **How it works**: It computes a probability score between 0 and 1 using the **Sigmoid function**:
  $$P(y=1|X) = \\frac{1}{1 + e^{-z}}$$
  where $z = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3$.
"""))

lr_cells.append(code_cell("""# Initialize the Logistic Regression model
model = LogisticRegression()
"""))

# 10. Model Training
lr_cells.append(code_cell("""# Fit the classification model
model.fit(X_train, y_train)
"""))

# 11. Predictions
lr_cells.append(code_cell("""# Generate outcome predictions (0 or 1)
predictions = model.predict(X_test)

# Compare values side-by-side
compare_df = pd.DataFrame({
    'Actual_Label': y_test,
    'Predicted_Label': predictions
})
print(compare_df.head())
"""))

# 12. Evaluation
lr_cells.append(md_cell("""# 12. Evaluation

Let's look at key classification metrics.
"""))

lr_cells.append(code_cell(get_classification_eval_code("Logistic Regression")))

lr_cells.append(md_cell("""### What Did We Observe?
* The model achieved excellent scores, with accuracy matching or exceeding 90%.

### What Did We Learn?
* Metrics like Precision and Recall help us measure trade-offs (e.g. false alarms vs missed failures).
"""))

# 13. Visualizing Performance
lr_cells.append(md_cell("""# 13. Visualizing Model Performance

We will plot:
1. **Confusion Matrix Heatmap**: Displays True Positives, True Negatives, False Positives, and False Negatives.
2. **Sigmoid Probability Curve**: Visualizing passing probability vs. hours studied.
"""))

lr_cells.append(code_cell("""# Plot 1: Confusion Matrix Heatmap
conf_matrix = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Fail', 'Predicted Pass'], 
            yticklabels=['Actual Fail', 'Actual Pass'])
plt.title('Logistic Regression Confusion Matrix')
plt.show()
"""))

lr_cells.append(code_cell("""# Plot 2: Sigmoid Probability Curve
# We isolate 'Hours_Studied' and fit a 1D model to visualize the Sigmoid curve clearly.
hours_only = df[['Hours_Studied']]
model_1d = LogisticRegression()
model_1d.fit(hours_only, y)

# Generate a dense range of study hours to plot smooth curve
dense_hours = np.linspace(0, 10, 300).reshape(-1, 1)
probs = model_1d.predict_proba(dense_hours)[:, 1]

plt.figure(figsize=(8, 4))
plt.scatter(df['Hours_Studied'], df['Pass_Fail'], color='red', alpha=0.5, label='Actual Data')
plt.plot(dense_hours, probs, color='blue', linewidth=3, label='Sigmoid Probability Curve')
plt.axhline(0.5, color='gray', linestyle='--', label='50% Threshold')
plt.title('Pass Probability Sigmoid Curve')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

lr_cells.append(md_cell("""### What Did We Observe?
* The curve forms a clear "S" shape.
* Students studying more than 3-4 hours cross the 50% probability line to passing.

### What Did We Learn?
* The Sigmoid curve translates input values into values between 0 and 1, representing probability.
"""))

# 14. Model Interpretation
lr_cells.append(code_cell("""# Coefficients and intercept
print("Intercept:", model.intercept_)
print("Coefficients:")
for col, coef in zip(X.columns, model.coef_[0]):
    print(f"* {col}: {coef:.4f}")
"""))

# 15. Conclusion
lr_cells.append(md_cell("""# 15. Conclusion
* We predicted pass/fail outcomes using study variables.
* The logistic curve cleanly maps study habits to pass probabilities.
"""))

# 16. Beginner Dictionary
lr_cells.append(md_cell(get_dictionary()))

make_notebook("models/Logistic_Regression.ipynb", lr_cells)

# =====================================================================
# 2. K-NEAREST NEIGHBORS (KNN)
# =====================================================================
knn_cells = []
knn_cells.append(md_cell("""# 1. Project Introduction

Welcome! In this notebook, we will explore **K-Nearest Neighbors (KNN)**, an intuitive and straightforward classification algorithm.

### What is KNN?
* It is a **supervised learning** classification algorithm.
* It operates on a very simple premise: **"Birds of a feather flock together."**
* To classify a new data point, the model looks at the $K$ closest data points in the training set (its neighbors) and takes a majority vote.

### Why does it exist?
* It is a non-parametric model (makes no assumptions about how the data is distributed), making it very flexible.

### Real-World Use Cases:
* **Recommendation Systems**: Suggesting movies similar to ones you have watched.
* **Image Recognition**: Grouping handwriting styles based on pixel distances.
"""))

knn_cells.append(md_cell("""# 2. Problem Statement

* **Goal**: Predict whether an athlete plays **Soccer (0)** or **Basketball (1)** based on their **Height** and **Weight**.
* **Business Value**: Helps sports academies automatically route incoming athletes to fitting athletic trials.
"""))

knn_cells.append(code_cell("""# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
"""))

knn_cells.append(md_cell("""# 4. Create Synthetic Dataset

We define height (in cm) and weight (in kg) for **100 athletes**.
* Basketball players are typically taller and heavier.
* Soccer players are typically shorter and lighter.
"""))

knn_cells.append(code_cell("""# Hardcoded lists of athlete body metrics
height = [
    170, 172, 174, 175, 176, 178, 180, 182, 183, 185, 188, 190, 192, 195, 198, 200, 202, 205, 208, 210,
    168, 171, 173, 176, 177, 179, 181, 184, 186, 189, 191, 193, 196, 199, 201, 203, 206, 209, 211, 213,
    169, 172, 175, 177, 178, 180, 182, 185, 187, 190, 193, 194, 197, 200, 202, 204, 207, 210, 212, 214,
    170, 173, 176, 178, 179, 181, 183, 186, 188, 191, 194, 195, 198, 201, 203, 205, 208, 211, 213, 215,
    165, 167, 170, 173, 175, 178, 182, 185, 188, 190, 193, 196, 198, 202, 205, 208, 210, 180, 190, 200
]

weight = [
    65,  68,  70,  72,  73,  75,  77,  80,  82,  85,  90,  93,  95,  98, 102, 105, 108, 112, 115, 118,
    63,  67,  69,  71,  74,  76,  78,  81,  83,  86,  91,  94,  96,  99, 103, 106, 109, 113, 116, 119,
    64,  66,  69,  73,  75,  77,  79,  82,  84,  87,  92,  93,  97, 100, 104, 107, 110, 114, 117, 120,
    65,  68,  70,  72,  75,  78,  80,  83,  85,  88,  93,  95,  98, 101, 105, 108, 111, 115, 118, 121,
    60,  62,  65,  68,  70,  73,  78,  82,  86,  89,  92,  95,  97, 101, 106, 109, 112, 85,  92,  100
]

sport = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1
]

df = pd.DataFrame({
    'Height_cm': height,
    'Weight_kg': weight,
    'Preferred_Sport': sport
})

print("Dataset Shape:", df.shape)
print("First 5 rows:")
print(df.head())
"""))

knn_cells.append(md_cell("""### What Did We Observe?
* The labels are `0` (Soccer) and `1` (Basketball).
* Features list has height (cm) and weight (kg).
"""))

# 5. EDA
knn_cells.append(md_cell("""# 5. Exploratory Data Analysis (EDA)
"""))

knn_cells.append(code_cell("""# Chart: Height vs Weight colored by Sport
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Height_cm', y='Weight_kg', hue='Preferred_Sport', data=df, palette='Set1', s=80)
plt.title('Height vs. Weight of Athletes')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

knn_cells.append(md_cell("""### What Did We Observe?
* A clear split: taller, heavier athletes (above ~185cm) play basketball, while shorter, lighter athletes play soccer.
* There is minor overlap in the middle.
"""))

# 6. Cleaning
knn_cells.append(code_cell("""# Data cleaning: Check for duplicates
print("Duplicates count:", df.duplicated().sum())
"""))

# 7. Feature Selection
knn_cells.append(code_cell("""# Feature Selection
X = df[['Height_cm', 'Weight_kg']]
y = df['Preferred_Sport']
"""))

# 8. Train-Test Split
knn_cells.append(code_cell("""# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""))

# 9. Model Building
knn_cells.append(md_cell("""# 9. Model Building

* **How it works conceptually**: To classify a point, we compute the distance (e.g. Euclidean distance) from that point to all other training points. We find the $K$ closest points. The class with the highest vote wins.
* **Why K=3?**: An odd number prevents ties in voting.
"""))

knn_cells.append(code_cell("""# Initialize KNN model with K=3
model = KNeighborsClassifier(n_neighbors=3)
"""))

# 10. Training
knn_cells.append(code_cell("""# Fit KNN model
model.fit(X_train, y_train)
"""))

# 11. Predictions
knn_cells.append(code_cell("""# Predict outcomes
predictions = model.predict(X_test)
"""))

# 12. Evaluation
knn_cells.append(code_cell(get_classification_eval_code("KNN")))

# 13. Visualizing Performance
knn_cells.append(md_cell("""# 13. Visualizing Model Performance

We will plot:
1. **Confusion Matrix Heatmap**.
2. **2D Decision Boundary**: Shows the classification territory established by the neighbors.
"""))

knn_cells.append(code_cell("""# Plot 1: Confusion Matrix Heatmap
conf_matrix = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Predicted Soccer', 'Predicted Basketball'], 
            yticklabels=['Actual Soccer', 'Actual Basketball'])
plt.title('KNN Confusion Matrix')
plt.show()
"""))

knn_cells.append(code_cell("""# Plot 2: Decision Boundary
# To draw the boundary, we create a mesh grid of height and weight values
x_min, x_max = X['Height_cm'].min() - 5, X['Height_cm'].max() + 5
y_min, y_max = X['Weight_kg'].min() - 5, X['Weight_kg'].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))

# Predict class labels for each grid cell
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = model.predict(grid_points)
grid_predictions = grid_predictions.reshape(xx.shape)

# Plot decision boundary region
plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, grid_predictions, alpha=0.3, cmap='Set1')
# Overlay actual scatter points
sns.scatterplot(x='Height_cm', y='Weight_kg', hue='Preferred_Sport', data=df, palette='Set1', edgecolor='black', s=80)
plt.title('KNN Decision Boundary (K=3)')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend(labels=['Soccer Region', 'Basketball Region', 'Soccer (Data)', 'Basketball (Data)'])
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

knn_cells.append(md_cell("""### What Did We Observe?
* The colored zones clearly illustrate where the model classifies athletes as Soccer players vs. Basketball players.
* The boundary captures the overlap area dynamically.

### What Did We Learn?
* The boundary line is non-linear. Changing $K$ changes the smoothness of the classification boundary.
"""))

# 14. Interpretation
knn_cells.append(md_cell("""# 14. Model Interpretation

Unlike Linear models, KNN is a **lazy learner** and does not have coefficients.
* **Neighbors**: Predictions are based strictly on distance coordinates.
* **Standardizing Data**: In real applications, features with larger scales (like height) can dominate features with smaller scales (like age). Scaling is highly recommended when metrics differ.
"""))

# 15. Conclusion
knn_cells.append(md_cell("""# 15. Conclusion
* We successfully categorized sports based on height and weight.
* KNN acts intuitively based on spatial proximity.
"""))

# 16. Beginner Dictionary
knn_cells.append(md_cell(get_dictionary()))

make_notebook("models/KNN.ipynb", knn_cells)

# =====================================================================
# 3. NAIVE BAYES
# =====================================================================
nb_cells = []
nb_cells.append(md_cell("""# 1. Project Introduction

Welcome! In this notebook, we will explore **Naive Bayes**, a probabilistic classifier built on Bayes' Theorem.

### What is Naive Bayes?
* It is a **supervised learning** classification algorithm.
* It calculates the probability of each class given the input features and selects the class with the highest probability.
* It is called **"Naive"** because it assumes that all input features are independent of one another. For example, it assumes a discount word in an email is independent of the email length, which is rarely true but simplifies calculations massively.

### Why does it exist?
* It is incredibly fast, simple to implement, and performs exceptionally well on text classification and spam filtering.

### Real-World Use Cases:
* **Spam Filters**: Identifying spam emails.
* **Sentiment Analysis**: Classifying review text as positive or negative.
"""))

nb_cells.append(md_cell("""# 2. Problem Statement

* **Goal**: Predict whether an incoming email is **Spam (1)** or **Not Spam (0)** based on its length and keywords.
* **Business Value**: Protects email inbox users from unwanted spam messages.
"""))

nb_cells.append(code_cell("""# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
"""))

nb_cells.append(md_cell("""# 4. Create Synthetic Dataset

We define email statistics for **100 emails**.
* **Email_Length**: Character count.
* **Contains_Discount_Word**: 1 if words like "free", "discount" are present, 0 otherwise.
* **Contains_Urgent_Word**: 1 if words like "now", "urgent" are present, 0 otherwise.
* **Is_Spam**: Target classification label.
"""))

nb_cells.append(code_cell("""# Hardcoded email dataset
length = [
    50,  120, 30,  300, 45,  15,  180, 220, 60,  400, 20,  150, 70,  500, 80,  90,  320, 250, 110, 420,
    40,  130, 35,  280, 55,  25,  190, 240, 65,  380, 15,  160, 75,  480, 85,  95,  310, 260, 115, 410,
    45,  140, 38,  290, 50,  28,  200, 230, 70,  390, 18,  170, 72,  490, 88,  98,  330, 270, 120, 430,
    42,  135, 32,  295, 52,  22,  185, 235, 68,  385, 12,  165, 78,  475, 82,  92,  315, 255, 112, 405,
    300, 320, 280, 340, 410, 150, 250, 190, 370, 290, 180, 220, 450, 480, 500, 90,  100, 120, 200, 300
]

contains_discount = [
    0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1,
    0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1,
    0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1,
    0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1
]

contains_urgent = [
    0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
    0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
    0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
    0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1
]

is_spam = [
    0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1,
    0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1,
    0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1,
    0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1
]

df = pd.DataFrame({
    'Email_Length': length,
    'Contains_Discount_Word': contains_discount,
    'Contains_Urgent_Word': contains_urgent,
    'Is_Spam': is_spam
})

print("Dataset Shape:", df.shape)
print("First 5 rows:")
print(df.head())
"""))

# 5. EDA
nb_cells.append(md_cell("""# 5. Exploratory Data Analysis (EDA)
"""))

nb_cells.append(code_cell("""# Chart 1: Email Length by Spam/Ham Class
plt.figure(figsize=(8, 4))
sns.boxplot(x='Is_Spam', y='Email_Length', data=df, palette='Pastel1')
plt.title('Email Length vs. Spam Class')
plt.xlabel('Is Spam (0 = Ham, 1 = Spam)')
plt.ylabel('Email Length (Characters)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
"""))

nb_cells.append(md_cell("""### What Did We Observe?
* Spam emails (class 1) generally have higher character counts than ham emails (class 0).
"""))

nb_cells.append(code_cell("""# Chart 2: Spam occurrence count by Contains_Discount_Word
plt.figure(figsize=(6, 4))
sns.countplot(x='Contains_Discount_Word', hue='Is_Spam', data=df, palette='Set2')
plt.title('Discount Words vs. Spam Class')
# Custom labels for readable charts
plt.xticks([0, 1], ['No Discount Word', 'Contains Discount Word'])
plt.xlabel('')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(['Not Spam', 'Spam'])
plt.show()
"""))

nb_cells.append(md_cell("""### What Did We Observe?
* The presence of a discount word correlates heavily with the Spam class. Very few non-spam emails contain discount words.
"""))

# 6. Cleaning
nb_cells.append(code_cell("""# Data cleaning check
print("Null values count:", df.isnull().sum().sum())
"""))

# 7. Feature Selection
nb_cells.append(code_cell("""# Feature Selection
X = df[['Email_Length', 'Contains_Discount_Word', 'Contains_Urgent_Word']]
y = df['Is_Spam']
"""))

# 8. Train-Test Split
nb_cells.append(code_cell("""# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""))

# 9. Model Building
nb_cells.append(md_cell("""# 9. Model Building

* **How it works conceptually**: It applies Bayes' Theorem:
  $$P(\text{Spam} | \text{Words}) = \\frac{P(\text{Words} | \text{Spam}) \times P(\text{Spam})}{P(\text{Words})}$$
  The algorithm multiplies the individual probability of each word occurring in a spam message, assuming word occurrences are independent.
"""))

nb_cells.append(code_cell("""# Initialize Gaussian Naive Bayes model
model = GaussianNB()
"""))

# 10. Training
nb_cells.append(code_cell("""# Fit model
model.fit(X_train, y_train)
"""))

# 11. Predictions
nb_cells.append(code_cell("""# Predict spam outcomes
predictions = model.predict(X_test)
"""))

# 12. Evaluation
nb_cells.append(code_cell(get_classification_eval_code("Naive Bayes")))

# 13. Visualizing Performance
nb_cells.append(md_cell("""# 13. Visualizing Model Performance

We will display:
1. **Confusion Matrix Heatmap**.
2. **Feature Probability Density Curves**: Plotting distribution of email length for spam vs ham.
"""))

nb_cells.append(code_cell("""# Plot 1: Confusion Matrix Heatmap
conf_matrix = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Predicted Ham', 'Predicted Spam'], 
            yticklabels=['Actual Ham', 'Actual Spam'])
plt.title('Naive Bayes Confusion Matrix')
plt.show()
"""))

nb_cells.append(code_cell("""# Plot 2: Probability Density Curves for Email_Length
plt.figure(figsize=(8, 4))
sns.kdeplot(df[df['Is_Spam'] == 0]['Email_Length'], label='Ham (Not Spam)', shade=True, color='blue')
sns.kdeplot(df[df['Is_Spam'] == 1]['Email_Length'], label='Spam', shade=True, color='red')
plt.title('Probability Density of Email Length')
plt.xlabel('Email Length')
plt.ylabel('Density Probability')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

nb_cells.append(md_cell("""### What Did We Observe?
* The peaks of the curves are distinct, indicating that email length is a useful factor for Gaussian probability estimation.
"""))

# 14. Model Interpretation
nb_cells.append(md_cell("""# 14. Model Interpretation

Naive Bayes calculates baseline prior probabilities:
* **Class Prior**: The overall probability of an email being spam ($P(\text{Spam})$) vs. ham ($P(\text{Ham})$) before reading its words.
"""))

nb_cells.append(code_cell("""# Display class priors calculated by model
print("Class Priors (Ham, Spam):", model.class_prior_)
print("Class Counts:", model.class_count_)
"""))

# 15. Conclusion
nb_cells.append(md_cell("""# 15. Conclusion
* We built a spam classifier using email features.
* Probabilistic classifiers are fast and perform exceptionally well under independence assumptions.
"""))

# 16. Dictionary
nb_cells.append(md_cell(get_dictionary()))

make_notebook("models/Logistic_Regression.ipynb", lr_cells)
make_notebook("models/KNN.ipynb", knn_cells)
make_notebook("models/Naive_Bayes.ipynb", nb_cells)
print("Finished basic classification notebooks generation!")
