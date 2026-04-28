import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 2. Load dataset (Iris)
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

print(r"C:\Users\namiy\Downloads\Groceries_dataset.csv (1).zip")
print(df.head())

# 3. Preprocessing
# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# 4. Standardize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

print("\n--- Data Standardized ---")

# 5. Apply PCA
pca = PCA()
pca_data = pca.fit_transform(scaled_data)

# 6. Explained variance ratio
explained_variance = pca.explained_variance_ratio_

print("\n--- Explained Variance Ratio ---")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.4f}")

# 7. Cumulative variance
cumulative_variance = np.cumsum(explained_variance)

print("\n--- Cumulative Variance ---")
for i, var in enumerate(cumulative_variance):
    print(f"PC{i+1}: {var:.4f}")

# Decide optimal components
optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
print("\nOptimal number of components (95% variance):", optimal_components)

# 8. Reduce to 2D
pca_2 = PCA(n_components=2)
reduced_data = pca_2.fit_transform(scaled_data)

print("\n--- Reduced Data (First 5 rows) ---")
print(reduced_data[:5])

# 9. Visualization

# Scree Plot
plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.title("Screen Plot")
plt.legend(["Variance"])
plt.show()

# Cumulative Variance Plot
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance")
plt.title("Cumulative Variance Plot")
plt.legend(["Cumulative Variance"])
plt.show()

# 2D Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(reduced_data[:,0], reduced_data[:,1], alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Projection")
plt.legend(["Data Points"])
plt.show()
