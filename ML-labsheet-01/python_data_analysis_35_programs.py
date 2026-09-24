import sys
print("Python version:", sys.version)


import numpy as np
print("NumPy version:", np.__version__)


import pandas as pd
print("Pandas version:", pd.__version__)


import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y, marker="o")
plt.title("Simple Line Plot")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.grid(True)
plt.show()

import seaborn as sns

sample_data = [12, 15, 14, 18, 20, 17, 16, 19, 21, 15]

sns.histplot(sample_data, kde=True)
plt.title("Basic Seaborn Statistical Plot")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

import sklearn
print("Scikit-learn version:", sklearn.__version__)


print("Hello, World!")
print("My first Python program in Jupyter Notebook.")



name = "Student"
print("Hello,", name)
print("Python script executed successfully.")


data = {
    "Name": ["Amit", "Priya", "Rahul", "Neha", "Vikas", "Amit"],
    "Age": [21, 22, 20, 23, 21, 21],
    "Marks": [85, 90, 78, 88, 76, 85],
    "City": ["Delhi", "Roorkee", "Haridwar", "Delhi", "Roorkee", "Delhi"]
}

df = pd.DataFrame(data)
df.to_csv("student_dataset.csv", index=False)

df = pd.read_csv("student_dataset.csv")
print("\nLoaded Dataset:")
print(df)

# ============================================================
# 12. Display first five records using head()
# ============================================================
print("\nFirst five records:")
print(df.head())

# ============================================================
# 13. Display last five records using tail()
# ============================================================
print("\nLast five records:")
print(df.tail())

# ============================================================
# 14. Find total number of rows and columns
# ============================================================
rows, columns = df.shape
print("\nNumber of rows:", rows)
print("Number of columns:", columns)

# ============================================================
# 15. Display names of all columns
# ============================================================
print("\nColumn names:")
print(df.columns.tolist())

# ============================================================
# 16. Check data types of all columns
# ============================================================
print("\nData types:")
print(df.dtypes)

# ============================================================
# 17. Generate descriptive statistics
# ============================================================
print("\nDescriptive statistics:")
print(df.describe())

# ============================================================
# 18. Display complete information using info()
# ============================================================
print("\nDataset information:")
df.info()

# ============================================================
# 19. Identify missing values
# ============================================================
print("\nMissing values:")
print(df.isnull())

# ============================================================
# 20. Count total missing values in each column
# ============================================================
print("\nTotal missing values in each column:")
print(df.isnull().sum())

# ============================================================
# 21. Display unique values in a selected column
# ============================================================
print("\nUnique cities:")
print(df["City"].unique())

# ============================================================
# 22. Count frequency of each unique value
# ============================================================
print("\nFrequency of each city:")
print(df["City"].value_counts())

# ============================================================
# 23. Rename one or more columns
# ============================================================
df_renamed = df.rename(columns={
    "Marks": "Score"
})

print("\nDataset after renaming column:")
print(df_renamed)

# ============================================================
# 24. Select specific rows and columns using loc[]
# ============================================================
print("\nUsing loc[]:")
print(df.loc[0:2, ["Name", "Marks"]])

# ============================================================
# 25. Select specific rows and columns using iloc[]
# ============================================================
print("\nUsing iloc[]:")
print(df.iloc[0:3, 0:3])

# ============================================================
# 26. Filter records based on a condition
# ============================================================
print("\nStudents with Marks greater than 80:")
filtered_df = df[df["Marks"] > 80]
print(filtered_df)

# ============================================================
# 27. Sort dataset using one or more columns
# ============================================================
print("\nDataset sorted by Marks:")
sorted_df = df.sort_values(by="Marks", ascending=False)
print(sorted_df)

# ============================================================
# 28. Add a new column
# ============================================================
df["Result"] = df["Marks"].apply(
    lambda marks: "Pass" if marks >= 40 else "Fail"
)

print("\nAfter adding Result column:")
print(df)

# ============================================================
# 29. Delete an existing column
# ============================================================
df_without_city = df.drop(columns=["City"])

print("\nAfter deleting City column:")
print(df_without_city)

# ============================================================
# 30. Remove duplicate records
# ============================================================
df_no_duplicates = df.drop_duplicates()

print("\nAfter removing duplicate records:")
print(df_no_duplicates)

# ============================================================
# 31. Save modified dataset as a new CSV file
# ============================================================
df_no_duplicates.to_csv("modified_student_dataset.csv", index=False)
print("\nModified dataset saved as modified_student_dataset.csv")

# ============================================================
# 32. Load a dataset directly from Scikit-learn
# ============================================================
from sklearn.datasets import load_iris

iris = load_iris()
iris_df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

print("\nScikit-learn Iris dataset:")
print(iris_df.head())

# ============================================================
# 33. Create a histogram for a numerical feature
# ============================================================
plt.figure(figsize=(7, 5))
plt.hist(iris_df["sepal length (cm)"], bins=10, edgecolor="black")
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# ============================================================
# 34. Create a scatter plot between two variables
# ============================================================
plt.figure(figsize=(7, 5))
plt.scatter(
    iris_df["sepal length (cm)"],
    iris_df["petal length (cm)"]
)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

# ============================================================
# 35. Generate correlation matrix and visualize using heatmap
# ============================================================
correlation_matrix = iris_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Matrix Heatmap")
plt.show()

print("\nAll 35 programs have been completed successfully.")
