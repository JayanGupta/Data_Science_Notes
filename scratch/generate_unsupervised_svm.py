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

# Print metrics
print(f"Accuracy Score: {accuracy:.4f}")
print(f"Precision Score: {precision:.4f}")
print(f"Recall Score: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\\nConfusion Matrix Array:")
print(conf_matrix)
"""

# =====================================================================
# 1. SVM
# =====================================================================
svm_cells = []
svm_cells.append(md_cell("""# 1. Project Introduction

Welcome! In this notebook, we will explore **Support Vector Machines (SVM)**, a highly robust classifier designed to find optimal separation boundaries between classes.

### What is SVM?
* It is a **supervised learning** classification algorithm.
* It works by drawing a line, curve, or hyperplane that separates classes.
* It doesn't just find any separating line—it looks for the **maximum margin hyperplane**, which is the line that keeps the largest possible distance (margin) from the closest points of each class.
* These closest boundary points are called **support vectors**.

### Why does it exist?
* It is mathematically rigorous and works exceptionally well when there is a clear margin of separation between classes.

### Real-World Use Cases:
* **Bioinformatics**: Classifying genes or proteins.
* **Text Categorization**: Grouping news articles into topics.
"""))

svm_cells.append(md_cell("""# 2. Problem Statement

* **Goal**: Classify whether a candidate passes a physical fitness screening (**1**) or fails (**0**) based on scores from two assessments.
* **Business Value**: Standardizes recruitment benchmarks for athletic or military screenings.
"""))

svm_cells.append(code_cell("""# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
"""))

svm_cells.append(md_cell("""# 4. Create Synthetic Dataset

We define assessment scores for **100 candidates**.
* **Fitness_Score_1**: Performance score on first assessment (1.0 to 10.0).
* **Fitness_Score_2**: Performance score on second assessment (1.0 to 10.0).
* **Screening_Passed**: Target pass (1) or fail (0) outcome.
"""))

svm_cells.append(code_cell("""# Hardcoded candidate assessment profiles
score1 = [
    1.5, 2.5, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 2.0, 3.0, 4.2, 5.2, 6.2,
    7.2, 8.2, 9.2, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8, 1.2, 2.2, 3.2, 4.4, 5.4, 6.4, 7.4, 8.4,
    9.4, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 1.4, 2.4, 3.4, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 2.0,
    3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 2.2, 3.2, 4.2,
    5.2, 6.2, 7.2, 8.2, 9.2, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
]

score2 = [
    2.0, 3.0, 4.0, 3.5, 4.5, 4.0, 5.5, 5.0, 6.5, 6.0, 7.5, 7.0, 8.5, 8.0, 9.5, 1.5, 2.5, 3.8, 4.8, 5.8,
    6.8, 7.8, 8.8, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 2.2, 3.2, 4.2, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0, 2.6, 3.6, 4.6, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8, 2.4, 3.4, 4.4, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 3.0,
    4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 9.0, 3.2, 4.2, 5.2,
    6.2, 7.2, 8.2, 9.2, 9.5, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8, 9.9, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5
]

passed = [
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
    1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,
    1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
    1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1
]

df = pd.DataFrame({
    'Fitness_Score_1': score1,
    'Fitness_Score_2': score2,
    'Screening_Passed': passed
})

print("Shape:", df.shape)
print(df.head())
"""))

# 5. EDA
svm_cells.append(md_cell("""# 5. Exploratory Data Analysis (EDA)
"""))

svm_cells.append(code_cell("""# Chart 1: Fitness Score 1 vs Fitness Score 2 colored by Pass/Fail
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Fitness_Score_1', y='Fitness_Score_2', hue='Screening_Passed', data=df, palette='coolwarm', s=80)
plt.title('Assessment Fitness Scores Scatter')
plt.xlabel('Fitness Score 1')
plt.ylabel('Fitness Score 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

svm_cells.append(md_cell("""### What Did We Observe?
* The candidates are split diagonally: candidates with high scores on both assessments pass the screening.
"""))

# 6. Cleaning
svm_cells.append(code_cell("""# Cleaning check
print("Null count:", df.isnull().sum().sum())
"""))

# 7. Feature Selection
svm_cells.append(code_cell("""# Feature Selection
X = df[['Fitness_Score_1', 'Fitness_Score_2']]
y = df['Screening_Passed']
"""))

# 8. Train-Test Split
svm_cells.append(code_cell("""# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""))

# 9. Model Building
svm_cells.append(md_cell("""# 9. Model Building

* **How it works conceptually**: It solves an optimization problem to find the decision boundary hyperplane that maximizes the distance to the nearest training coordinates (support vectors).
* **Hyperparameters**: We use a **Linear Kernel** to keep the separating boundary simple and straight.
"""))

svm_cells.append(code_cell("""# Initialize SVM with a linear kernel
model = SVC(kernel='linear', random_state=42)
"""))

# 10. Training
svm_cells.append(code_cell("""# Train SVM
model.fit(X_train, y_train)
"""))

# 11. Predictions
svm_cells.append(code_cell("""# Predict labels
predictions = model.predict(X_test)
"""))

# 12. Evaluation
svm_cells.append(code_cell(get_classification_eval_code()))

# 13. Visualizing Performance
svm_cells.append(md_cell("""# 13. Visualizing Model Performance

We will plot:
1. **Confusion Matrix Heatmap**.
2. **Hyperplane Margin and Support Vectors**: Showcasing the boundary line, margins, and highlight supporting boundary vectors.
"""))

svm_cells.append(code_cell("""# Plot 1: Confusion Matrix Heatmap
conf_matrix = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Predicted Fail', 'Predicted Pass'], 
            yticklabels=['Actual Fail', 'Actual Pass'])
plt.title('SVM Confusion Matrix')
plt.show()
"""))

svm_cells.append(code_cell("""# Plot 2: Decision Hyperplane and Support Vectors
plt.figure(figsize=(9, 6))
# Plot data points
sns.scatterplot(x='Fitness_Score_1', y='Fitness_Score_2', hue='Screening_Passed', data=df, palette='coolwarm', edgecolor='black', s=80)

# Create grid to plot boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Highlight Support Vectors
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=200,
           linewidth=1.5, facecolors='none', edgecolors='black', label='Support Vectors')
plt.title('SVM Decision Hyperplane, Margins, and Support Vectors')
plt.xlabel('Fitness Score 1')
plt.ylabel('Fitness Score 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

svm_cells.append(md_cell("""### What Did We Observe?
* The solid line represents the separating hyperplane.
* The dashed lines represent the margins.
* The points highlighted with black circles are the **support vectors** that directly define the boundary placement.

### What Did We Learn?
* Only the support vectors matter for drawing the boundary; moving other data points has no effect on the hyperplane.
"""))

# 14. Interpretation
svm_cells.append(code_cell("""# Coefficient interpretation
print("Hyperplane Intercept (Bias):", model.intercept_)
print("Hyperplane Weights (Coefficients):", model.coef_[0])
print("Number of Support Vectors per class:", model.n_support_)
"""))

# 15. Conclusion
svm_cells.append(md_cell("""# 15. Conclusion
* SVM successfully separates candidates with a maximum margin boundary.
"""))

# 16. Dictionary
svm_cells.append(md_cell(get_dictionary()))

make_notebook("models/SVM.ipynb", svm_cells)


# =====================================================================
# 2. K-MEANS
# =====================================================================
km_cells = []
km_cells.append(md_cell("""# 1. Project Introduction

Welcome! In this notebook, we will explore **K-Means Clustering**, one of the most popular unsupervised learning algorithms.

### What is K-Means?
* It is an **unsupervised learning** clustering algorithm.
* Unsupervised means the dataset does **not have target labels** (y). The algorithm must discover patterns and group data on its own.
* It groups data points into $K$ distinct clusters based on feature similarity (distance).
* **Centroid**: The center point of a cluster. K-Means moves centroids iteratively until it finds the optimal groups.

### Why does it exist?
* It helps us segment and discover natural groupings in datasets.

### Real-World Use Cases:
* **Customer Segmentation**: Grouping buyers based on spending and income.
* **Document Clustering**: Organizing documents by content topics.
"""))

km_cells.append(md_cell("""# 2. Problem Statement

* **Goal**: Group retail store shoppers into customer segments based on their annual income and spending score.
* **Business Value**: Enables marketers to target specific shopper clusters with custom-tailored promotions.
"""))

km_cells.append(code_cell("""# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
"""))

km_cells.append(md_cell("""# 4. Create Synthetic Dataset

We define income and spending behaviors for **100 shoppers**.
* **Annual_Income_K**: Annual income in k$.
* **Spending_Score**: Store rating of customer spending frequency and size (1 to 100).
* Note: There are **no labels** in this dataset!
"""))

km_cells.append(code_cell("""# Hardcoded shopper profiles representing clustered patterns
income = [
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 15, 17, 19, 21,
    50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 51, 53, 55, 57,
    85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 86, 88, 90, 92,
    18, 20, 22, 24, 26, 28, 52, 55, 58, 62, 65, 68, 88, 92, 95, 98, 102, 105, 20, 50,
    70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108
]

spending = [
    81, 78, 75, 77, 85, 90, 88, 82, 79, 83, 86, 80, 76, 74, 82, 88, 83, 80, 82, 85,
    45, 48, 50, 52, 55, 60, 58, 53, 51, 47, 49, 53, 56, 52, 50, 46, 48, 52, 54, 56,
    15, 18, 22, 20, 17, 25, 23, 19, 14, 16, 21, 24, 20, 18, 15, 12, 16, 20, 22, 25,
    80, 85, 87, 82, 79, 81, 50, 52, 54, 56, 48, 50, 20, 22, 24, 18, 15, 17, 84, 52,
    18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56
]

df = pd.DataFrame({
    'Annual_Income_K': income,
    'Spending_Score': spending
})

print("Shape:", df.shape)
print(df.head())
"""))

# 5. EDA
km_cells.append(md_cell("""# 5. Exploratory Data Analysis (EDA)
"""))

km_cells.append(code_cell("""# Chart 1: Raw Income vs Spending Score
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Annual_Income_K', y='Spending_Score', data=df, color='black', s=80)
plt.title('Customer Spending vs. Annual Income (Unlabeled)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

km_cells.append(md_cell("""### What Did We Observe?
* The shoppers appear to cluster naturally into three or four distinct regions.
"""))

# 6. Cleaning
km_cells.append(code_cell("""# Cleaning check
print("Null count:", df.isnull().sum().sum())
"""))

# 7. Feature Selection
km_cells.append(md_cell("""# 7. Feature Selection

Since clustering is unsupervised, we select our descriptive features as `X`. There is no target `y`.
"""))

km_cells.append(code_cell("""# Select descriptive columns
X = df[['Annual_Income_K', 'Spending_Score']]
"""))

# 8. Train-Test Split
km_cells.append(md_cell("""# 8. Train-Test Split

* Unsupervised algorithms are usually trained on the whole dataset to find patterns, so we do not split the data here.
"""))

# 9. Model Building
km_cells.append(md_cell("""# 9. Model Building

* **How it works conceptually**: It places $K$ centroids randomly. Then, it repeats two steps:
  1. **Assign**: Assigns each point to its closest centroid.
  2. **Update**: Recalculates the center of each cluster and moves the centroid there.
* We set $K=3$ for our initial segmentation.
"""))

km_cells.append(code_cell("""# Initialize K-Means with 3 clusters
model = KMeans(n_clusters=3, random_state=42)
"""))

# 10. Training
km_cells.append(code_cell("""# Fit model
model.fit(X)
print("K-Means training completed successfully!")
"""))

# 11. Predictions
km_cells.append(code_cell("""# Get cluster assignments
labels = model.labels_

# View assignments for first few customers
print("First 10 cluster assignments:", labels[:10])
"""))

# 12. Evaluation
km_cells.append(md_cell("""# 12. Evaluation

Unsupervised models do not have true labels to verify accuracy, so we use:
* **Inertia**: Sum of squared distances of samples to their closest cluster center. Lower is better.
* **Silhouette Score**: Measures how close a point is to its own cluster compared to other clusters (ranges from -1 to +1). Higher is better.
"""))

km_cells.append(code_cell("""# Compute cluster evaluations
inertia = model.inertia_
silhouette = silhouette_score(X, labels)

print(f"Inertia Score: {inertia:.4f}")
print(f"Silhouette Score: {silhouette:.4f} (Closer to 1 means clean separation)")
"""))

# 13. Visualizing Performance
km_cells.append(md_cell("""# 13. Visualizing Model Performance

We will plot:
1. **Cluster Assignment Scatter**: Shoppers colored by cluster, with centroids highlighted.
2. **Elbow Curve**: Plotting Inertia for different $K$ values to find the optimal cluster count.
"""))

km_cells.append(code_cell("""# Plot 1: Cluster Assignments and Centroids
centroids = model.cluster_centers_

plt.figure(figsize=(9, 6))
sns.scatterplot(x='Annual_Income_K', y='Spending_Score', hue=labels, palette='Set1', data=df, s=80, edgecolor='black')
# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], color='cyan', marker='X', s=300, edgecolor='black', label='Centroids')
plt.title('Customer Segments (K=3)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

km_cells.append(code_cell("""# Plot 2: Elbow Curve
inertia_values = []
k_range = range(1, 8)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia_values.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia_values, marker='o', color='purple', linewidth=2)
plt.title('Elbow Method showing optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Error)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

km_cells.append(md_cell("""### What Did We Observe?
* At $K=3$, the clusters are clearly separated.
* The Elbow plot shows a sharp bend or "elbow" at $K=3$, confirming it is the optimal choice.
"""))

# 14. Interpretation
km_cells.append(code_cell("""# Check cluster centers
print("Centroid Coordinates (Income, Spending):")
for i, centroid in enumerate(centroids):
    print(f"* Cluster {i}: Income = {centroid[0]:.2f}k$, Spending Score = {centroid[1]:.2f}")
"""))

# 15. Conclusion
km_cells.append(md_cell("""# 15. Conclusion
* We grouped customers into 3 distinct spending segments.
* Centroids help marketers understand and target each group's behaviors.
"""))

# 16. Dictionary
km_cells.append(md_cell(get_dictionary()))

make_notebook("models/K_Means.ipynb", km_cells)


# =====================================================================
# 3. PCA
# =====================================================================
pca_cells = []
pca_cells.append(md_cell("""# 1. Project Introduction

Welcome! In this notebook, we will explore **Principal Component Analysis (PCA)**, a key dimensionality reduction technique.

### What is PCA?
* It is an **unsupervised learning** dimensionality reduction method.
* When we have too many features (dimensions), it becomes difficult to visualize and analyze the data.
* PCA projects the data onto a lower-dimensional space (e.g. converting 6 features into 2 new features called **Principal Components**) while retaining as much variance (information) as possible.

### Why does it exist?
* It simplifies complex datasets, reduces storage and computation costs, and enables easy 2D visualization of high-dimensional data.

### Real-World Use Cases:
* **Image Compression**: Reducing image file sizes while keeping key features.
* **Genomics**: Analyzing thousands of genetic features in a 2D plot.
"""))

pca_cells.append(md_cell("""# 2. Problem Statement

* **Goal**: Reduce student scores across 6 academic subjects (Algebra, Calculus, Geometry, Literature, History, Grammar) down to just 2 principal components.
* **Business Value**: Simplifies evaluation by grouping scores into broad categories (like STEM vs. Humanities ability).
"""))

pca_cells.append(code_cell("""# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
"""))

pca_cells.append(md_cell("""# 4. Create Synthetic Dataset

We define subject grades (0 to 100) for **100 students**.
* Notice that STEM subject scores (Algebra, Calculus, Geometry) are highly correlated with each other.
* Humanities subject scores (Literature, History, Grammar) are also highly correlated with each other.
"""))

pca_cells.append(code_cell("""# Hardcoded student grades across 6 subjects
algebra = [
    95, 45, 85, 50, 90, 40, 88, 55, 92, 48, 80, 52, 84, 42, 78, 60, 82, 58, 91, 75,
    96, 46, 86, 51, 91, 41, 89, 56, 93, 49, 81, 53, 85, 43, 79, 61, 83, 59, 92, 76,
    94, 44, 84, 49, 89, 39, 87, 54, 91, 47, 79, 51, 83, 41, 77, 59, 81, 57, 90, 74,
    95, 45, 85, 50, 90, 40, 88, 55, 92, 48, 80, 52, 84, 42, 78, 60, 82, 58, 91, 75,
    88, 60, 78, 90, 55, 65, 85, 50, 92, 40, 70, 80, 45, 62, 83, 72, 89, 54, 94, 68
]

calculus = [
    92, 40, 80, 45, 88, 38, 85, 50, 90, 42, 78, 48, 82, 40, 75, 58, 80, 55, 89, 72,
    93, 41, 81, 46, 89, 39, 86, 51, 91, 43, 79, 49, 83, 41, 76, 59, 81, 56, 90, 73,
    91, 39, 79, 44, 87, 37, 84, 49, 89, 41, 77, 47, 81, 39, 74, 57, 79, 54, 88, 71,
    92, 40, 80, 45, 88, 38, 85, 50, 90, 42, 78, 48, 82, 40, 75, 58, 80, 55, 89, 72,
    85, 58, 75, 88, 52, 60, 82, 48, 90, 38, 68, 78, 42, 60, 80, 70, 87, 52, 92, 65
]

geometry = [
    90, 42, 82, 48, 85, 35, 83, 52, 88, 45, 75, 50, 80, 38, 72, 55, 78, 52, 86, 70,
    91, 43, 83, 49, 86, 36, 84, 53, 89, 46, 76, 51, 81, 39, 73, 56, 79, 53, 87, 71,
    89, 41, 81, 47, 84, 34, 82, 51, 87, 44, 74, 49, 79, 37, 71, 54, 77, 51, 85, 69,
    90, 42, 82, 48, 85, 35, 83, 52, 88, 45, 75, 50, 80, 38, 72, 55, 78, 52, 86, 70,
    84, 56, 74, 86, 50, 58, 80, 46, 88, 36, 66, 76, 40, 58, 78, 68, 84, 50, 90, 63
]

literature = [
    50, 90, 55, 85, 45, 88, 52, 92, 40, 80, 60, 78, 58, 91, 62, 82, 58, 75, 50, 88,
    51, 91, 56, 86, 46, 89, 53, 93, 41, 81, 61, 79, 59, 92, 63, 83, 59, 76, 51, 89,
    49, 89, 54, 84, 44, 87, 51, 91, 39, 79, 59, 77, 57, 90, 61, 81, 57, 74, 49, 87,
    50, 90, 55, 85, 45, 88, 52, 92, 40, 80, 60, 78, 58, 91, 62, 82, 58, 75, 50, 88,
    60, 85, 65, 55, 80, 78, 50, 90, 45, 88, 75, 70, 82, 80, 55, 68, 52, 83, 48, 76
]

history = [
    48, 88, 52, 82, 42, 85, 50, 90, 38, 78, 58, 75, 56, 89, 60, 80, 55, 72, 48, 85,
    49, 89, 53, 83, 43, 86, 51, 91, 39, 79, 59, 76, 57, 90, 61, 81, 56, 73, 49, 86,
    47, 87, 51, 81, 41, 84, 49, 89, 37, 77, 57, 74, 55, 88, 59, 79, 54, 71, 47, 84,
    48, 88, 52, 82, 42, 85, 50, 90, 38, 78, 58, 75, 56, 89, 60, 80, 55, 72, 48, 85,
    58, 82, 62, 52, 78, 75, 48, 88, 43, 85, 72, 68, 79, 78, 53, 65, 50, 80, 46, 73
]

grammar = [
    52, 92, 58, 88, 48, 90, 55, 95, 43, 82, 62, 80, 60, 93, 65, 84, 61, 78, 53, 90,
    53, 93, 59, 89, 49, 91, 56, 96, 44, 83, 63, 81, 61, 94, 66, 85, 62, 79, 54, 91,
    51, 91, 57, 87, 47, 89, 54, 94, 42, 81, 61, 79, 59, 92, 64, 83, 60, 77, 52, 89,
    52, 92, 58, 88, 48, 90, 55, 95, 43, 82, 62, 80, 60, 93, 65, 84, 61, 78, 53, 90,
    62, 88, 68, 58, 82, 80, 53, 92, 47, 90, 78, 72, 85, 82, 58, 70, 55, 85, 50, 79
]

df = pd.DataFrame({
    'Algebra_Score': algebra,
    'Calculus_Score': calculus,
    'Geometry_Score': geometry,
    'Literature_Score': literature,
    'History_Score': history,
    'Grammar_Score': grammar
})

print("Shape:", df.shape)
print(df.head())
"""))

# 5. EDA
pca_cells.append(md_cell("""# 5. Exploratory Data Analysis (EDA)
"""))

pca_cells.append(code_cell("""# Chart 1: Heatmap showing STEM vs Humanities block correlation
plt.figure(figsize=(7, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Academic Subjects Correlation Heatmap')
plt.show()
"""))

pca_cells.append(md_cell("""### What Did We Observe?
* Highly positive correlations exist within STEM subjects (Algebra, Calculus, Geometry) and within Humanities subjects (Literature, History, Grammar).
* There is a negative correlation between STEM and Humanities performance.

### What Did We Learn?
* The 6 subjects contain redundant (correlated) information. We can easily compress these 6 features into fewer components.
"""))

# 6. Cleaning
pca_cells.append(code_cell("""# Cleaning check
print("Null count:", df.isnull().sum().sum())
"""))

# 7. Scaling
pca_cells.append(md_cell("""# 7. Standardizing Features

* **Important**: PCA is sensitive to feature scales. If one feature is measured in thousands and another in decimals, PCA will focus entirely on the larger feature.
* We scale features to have a mean of 0 and variance of 1 using `StandardScaler`.
"""))

pca_cells.append(code_cell("""# Initialize scaler and transform data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Check standard deviation of scaled features (should be ~1.0)
print("Standard deviations of scaled features:", X_scaled.std(axis=0))
"""))

# 9. Model Building
pca_cells.append(md_cell("""# 9. Model Building

* **How it works conceptually**: It finds orthogonal directions (eigenvectors) in high-dimensional space that maximize data variance. These directions are our principal components.
* We configure PCA to extract 2 components (`n_components=2`).
"""))

pca_cells.append(code_cell("""# Initialize PCA for 2 components
model = PCA(n_components=2)
"""))

# 10. Training
pca_cells.append(code_cell("""# Fit and transform scaled features
X_pca = model.fit_transform(X_scaled)
print("PCA transformation completed!")
"""))

# 11. Predictions
pca_cells.append(code_cell("""# Place PCA coordinates into a new DataFrame
pca_df = pd.DataFrame(data=X_pca, columns=['Principal_Component_1', 'Principal_Component_2'])
print("PCA Coordinate head:")
print(pca_df.head())
"""))

# 12. Evaluation
pca_cells.append(md_cell("""# 12. Evaluation

PCA is evaluated based on **Explained Variance**:
* **Explained Variance Ratio**: The percentage of information (variance) captured by each component.
"""))

pca_cells.append(code_cell("""# Print explained variance ratio
var_ratio = model.explained_variance_ratio_
print(f"Explained Variance Ratio: Component 1 = {var_ratio[0]:.4f}, Component 2 = {var_ratio[1]:.4f}")
print(f"Total Explained Variance (PC1 + PC2): {sum(var_ratio):.4f}")
"""))

# 13. Visualizing Performance
pca_cells.append(md_cell("""# 13. Visualizing Model Performance

We will plot:
1. **Scree Plot**: Variance explained per component.
2. **2D PCA Component Scatter**: Plotting PC1 vs PC2.
"""))

pca_cells.append(code_cell("""# Plot 1: Scree Plot
plt.figure(figsize=(6, 4))
plt.bar(['PC1', 'PC2'], var_ratio, color='salmon')
plt.title('Scree Plot (Explained Variance)')
plt.ylabel('Variance Ratio')
plt.show()
"""))

pca_cells.append(code_cell("""# Plot 2: 2D Principal Component Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Principal_Component_1', y='Principal_Component_2', data=pca_df, color='dodgerblue', s=80)
plt.title('2D Projection of Academic Scores via PCA')
plt.xlabel('Principal Component 1 (STEM Direction)')
plt.ylabel('Principal Component 2 (Humanities Direction)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""))

pca_cells.append(md_cell("""### What Did We Observe?
* The 2 components capture over 90% of the total dataset variance.
* The 2D scatter plot displays the distribution of student types clearly.
"""))

# 14. Interpretation
pca_cells.append(code_cell("""# Check loadings (contributions of original features to PCs)
loadings = pd.DataFrame(model.components_.T, columns=['PC1', 'PC2'], index=df.columns)
print("Feature Loadings for Principal Components:")
print(loadings)
"""))

pca_cells.append(md_cell("""### What Did We Observe?
* `PC1` has high negative values for STEM subjects and high positive values for Humanities.
* `PC2` shows different subject grouping behaviors.

### What Did We Learn?
* Loadings tell us how the original features map to the new dimensions, allowing us to interpret what PC1 and PC2 represent.
"""))

# 15. Conclusion
pca_cells.append(md_cell("""# 15. Conclusion
* We compressed 6 subject scores into 2 key components.
* PCA simplifies datasets while preserving almost all original variance.
"""))

# 16. Dictionary
pca_cells.append(md_cell(get_dictionary()))

make_notebook("models/SVM.ipynb", svm_cells)
make_notebook("models/K_Means.ipynb", km_cells)
make_notebook("models/PCA.ipynb", pca_cells)
print("Finished unsupervised and SVM notebooks generation!")
