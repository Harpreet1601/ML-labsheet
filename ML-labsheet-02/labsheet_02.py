import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    LabelEncoder,
)
from sklearn.feature_selection import SelectKBest, f_regression

# ============================================================
# DATASET SETUP
# ============================================================

CSV_FILE = "labsheet_02.csv"

# Create a sample dataset for practice if the CSV file does not exist.
if not os.path.exists(CSV_FILE):
    sample_data = {
        "Name": [
            "Amit", "Priya", "Rahul", "Neha", "Vikas",
            "Riya", "Karan", "Sneha", "Arjun", "Pooja",
            "Amit", "Rohan", "Anjali", "Mohit", "Simran"
        ],
        "Age": [
            21, 22, np.nan, 23, 21,
            24, 22, np.nan, 25, 23,
            21, 22, 24, 23, np.nan
        ],
        "Income": [
            25000, 30000, 28000, np.nan, 32000,
            35000, 29000, 31000, 1000000, 33000,
            np.nan, 36000, 34000, 27000, 38000
        ],
        "Marks": [
            85, 90, 78, 88, np.nan,
            92, 76, 89, 95, 84,
            85, np.nan, 91, 79, 87
        ],
        "City": [
            "Delhi", "Roorkee", "Haridwar", "Delhi", np.nan,
            "Roorkee", "Haridwar", "Delhi", "Roorkee", "Delhi",
            "Delhi", "Haridwar", np.nan, "Roorkee", "Delhi"
        ],
        "Gender": [
            "Male", "Female", "Male", "Female", "Male",
            "Female", "Male", "Female", "Male", "Female",
            "Male", "Male", "Female", "Male", "Female"
        ],
        "Date": [
            "2025-01-10", "2025-02-15", "2025-03-20", "2025-04-05",
            "2025-05-12", "2025-06-18", "2025-07-21", "2025-08-09",
            "2025-09-14", "2025-10-22", "2025-11-11", "2025-12-05",
            "2026-01-17", "2026-02-20", "2026-03-25"
        ]
    }

    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv(CSV_FILE, index=False)
    print(f"Sample dataset created: {CSV_FILE}")

# ============================================================
# 1. Load a dataset and identify missing values in each column
# ============================================================

df = pd.read_csv(CSV_FILE)

print("\n1. Missing values in each column:")
print(df.isnull().sum())

# Keep an original copy for comparisons
original_df = df.copy()

# ============================================================
# 2. Display percentage of missing values in every feature
# ============================================================

missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\n2. Percentage of missing values:")
print(missing_percentage)

# ============================================================
# 3. Remove rows containing missing values
# ============================================================

df_drop_rows = df.dropna()

print("\n3. Dataset after removing rows containing missing values:")
print(df_drop_rows)

# ============================================================
# 4. Remove columns having more than 50% missing values
# ============================================================

threshold = 50
columns_to_remove = missing_percentage[
    missing_percentage > threshold
].index

df_drop_columns = df.drop(columns=columns_to_remove)

print("\n4. Columns having more than 50% missing values:")
print(list(columns_to_remove))

print("\nDataset after removing those columns:")
print(df_drop_columns)

# ============================================================
# 5. Replace missing numerical values using the mean
# ============================================================

df_mean = df.copy()

numeric_columns = df_mean.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    df_mean[column] = df_mean[column].fillna(
        df_mean[column].mean()
    )

print("\n5. Missing numerical values replaced using mean:")
print(df_mean)

# ============================================================
# 6. Replace missing numerical values using the median
# ============================================================

df_median = df.copy()

for column in numeric_columns:
    df_median[column] = df_median[column].fillna(
        df_median[column].median()
    )

print("\n6. Missing numerical values replaced using median:")
print(df_median)

# ============================================================
# 7. Replace missing categorical values using the mode
# ============================================================

df_mode = df.copy()

categorical_columns = df_mode.select_dtypes(
    exclude=np.number
).columns

for column in categorical_columns:
    if not df_mode[column].mode().empty:
        df_mode[column] = df_mode[column].fillna(
            df_mode[column].mode()[0]
        )

print("\n7. Missing categorical values replaced using mode:")
print(df_mode)

# ============================================================
# 8. Fill missing values using forward fill
# ============================================================

df_forward = df.copy()
df_forward = df_forward.ffill()

print("\n8. Forward fill:")
print(df_forward)

# ============================================================
# 9. Fill missing values using backward fill
# ============================================================

df_backward = df.copy()
df_backward = df_backward.bfill()

print("\n9. Backward fill:")
print(df_backward)

# ============================================================
# 10. Compare dataset before and after handling missing values
# ============================================================

comparison_missing = pd.DataFrame({
    "Before Missing Values": original_df.isnull().sum(),
    "After Median/Mode": df_median.isnull().sum()
})

print("\n10. Comparison before and after missing-value treatment:")
print(comparison_missing)

# ============================================================
# Prepare numerical data for outlier operations
# ============================================================

outlier_df = df.copy()

# Fill missing numerical values with median for outlier analysis
for column in numeric_columns:
    outlier_df[column] = outlier_df[column].fillna(
        outlier_df[column].median()
    )

# ============================================================
# 11. Detect outliers using IQR method
# ============================================================

print("\n11. Outliers using IQR method:")

iqr_outliers = {}

for column in numeric_columns:
    Q1 = outlier_df[column].quantile(0.25)
    Q3 = outlier_df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = (
        (outlier_df[column] < lower_bound)
        | (outlier_df[column] > upper_bound)
    )

    iqr_outliers[column] = outlier_df.loc[mask, column].tolist()

print(iqr_outliers)

# ============================================================
# 12. Detect outliers using Z-score method
# ============================================================

print("\n12. Outliers using Z-score method:")

zscore_outliers = {}

for column in numeric_columns:
    mean = outlier_df[column].mean()
    std = outlier_df[column].std()

    if std != 0:
        z_scores = (
            (outlier_df[column] - mean) / std
        )

        zscore_outliers[column] = outlier_df.loc[
            z_scores.abs() > 3, column
        ].tolist()
    else:
        zscore_outliers[column] = []

print(zscore_outliers)

# ============================================================
# 13. Visualize outliers using Box Plot
# ============================================================

plt.figure(figsize=(10, 6))
sns.boxplot(data=outlier_df[numeric_columns])
plt.title("13. Box Plot for Outlier Detection")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================================================
# 14. Visualize outliers using Scatter Plot
# ============================================================

if len(numeric_columns) >= 2:
    x_column = numeric_columns[0]
    y_column = numeric_columns[1]

    plt.figure(figsize=(8, 5))
    plt.scatter(
        outlier_df[x_column],
        outlier_df[y_column]
    )
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(
        f"14. Scatter Plot: {x_column} vs {y_column}"
    )
    plt.show()

# ============================================================
# 15. Remove outliers using IQR method
# ============================================================

df_iqr_removed = outlier_df.copy()

for column in numeric_columns:
    Q1 = df_iqr_removed[column].quantile(0.25)
    Q3 = df_iqr_removed[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_iqr_removed = df_iqr_removed[
        (df_iqr_removed[column] >= lower_bound)
        & (df_iqr_removed[column] <= upper_bound)
    ]

print("\n15. Dataset after removing IQR outliers:")
print(df_iqr_removed)

# ============================================================
# 16. Replace outliers with the median value
# ============================================================

df_outlier_median = outlier_df.copy()

for column in numeric_columns:
    Q1 = df_outlier_median[column].quantile(0.25)
    Q3 = df_outlier_median[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    median_value = df_outlier_median[column].median()

    mask = (
        (df_outlier_median[column] < lower_bound)
        | (df_outlier_median[column] > upper_bound)
    )

    df_outlier_median.loc[mask, column] = median_value

print("\n16. Outliers replaced with median:")
print(df_outlier_median)

# ============================================================
# 17. Cap outliers using percentile-based capping
# ============================================================

df_capped = outlier_df.copy()

for column in numeric_columns:
    lower_limit = df_capped[column].quantile(0.01)
    upper_limit = df_capped[column].quantile(0.99)

    df_capped[column] = df_capped[column].clip(
        lower_limit,
        upper_limit
    )

print("\n17. Dataset after percentile-based capping:")
print(df_capped)

# ============================================================
# 18. Compare dataset before and after outlier treatment
# ============================================================

outlier_comparison = pd.DataFrame({
    "Before": outlier_df[numeric_columns].mean(),
    "After Median Replacement":
        df_outlier_median[numeric_columns].mean()
})

print("\n18. Comparison before and after outlier treatment:")
print(outlier_comparison)

# ============================================================
# 19. Apply Min-Max Normalization
# ============================================================

minmax_scaler = MinMaxScaler()

df_minmax = outlier_df.copy()

df_minmax[numeric_columns] = minmax_scaler.fit_transform(
    df_minmax[numeric_columns]
)

print("\n19. Min-Max Normalization:")
print(df_minmax.head())

# ============================================================
# 20. Apply Standardization (Z-score Scaling)
# ============================================================

standard_scaler = StandardScaler()

df_standard = outlier_df.copy()

df_standard[numeric_columns] = standard_scaler.fit_transform(
    df_standard[numeric_columns]
)

print("\n20. Standardization:")
print(df_standard.head())

# ============================================================
# 21. Apply Robust Scaling
# ============================================================

robust_scaler = RobustScaler()

df_robust = outlier_df.copy()

df_robust[numeric_columns] = robust_scaler.fit_transform(
    df_robust[numeric_columns]
)

print("\n21. Robust Scaling:")
print(df_robust.head())

# ============================================================
# 22. Apply Max Absolute Scaling
# ============================================================

maxabs_scaler = MaxAbsScaler()

df_maxabs = outlier_df.copy()

df_maxabs[numeric_columns] = maxabs_scaler.fit_transform(
    df_maxabs[numeric_columns]
)

print("\n22. Max Absolute Scaling:")
print(df_maxabs.head())

# ============================================================
# 23. Compare original and normalized datasets
# ============================================================

print("\n23. Original numerical data:")
print(outlier_df[numeric_columns].head())

print("\nMin-Max normalized data:")
print(df_minmax[numeric_columns].head())

# ============================================================
# 24. Visualize effect of normalization using histograms
# ============================================================

feature = numeric_columns[0]

plt.figure(figsize=(8, 5))
plt.hist(
    outlier_df[feature],
    bins=10,
    alpha=0.6,
    label="Original"
)
plt.hist(
    df_minmax[feature],
    bins=10,
    alpha=0.6,
    label="Min-Max"
)
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.title("24. Original vs Min-Max Normalized Data")
plt.legend()
plt.show()

# ============================================================
# 25. Visualize effect of scaling using box plots
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].boxplot(
    outlier_df[numeric_columns].values
)
axes[0].set_title("Original Data")
axes[0].set_xticks(
    range(1, len(numeric_columns) + 1),
    numeric_columns,
    rotation=45
)

axes[1].boxplot(
    df_standard[numeric_columns].values
)
axes[1].set_title("Standardized Data")
axes[1].set_xticks(
    range(1, len(numeric_columns) + 1),
    numeric_columns,
    rotation=45
)

plt.tight_layout()
plt.show()

# ============================================================
# 26. Compare different scaling techniques
# ============================================================

scaling_comparison = pd.DataFrame({
    "Original": outlier_df[feature].values,
    "MinMax": df_minmax[feature].values,
    "Standard": df_standard[feature].values,
    "Robust": df_robust[feature].values,
    "MaxAbs": df_maxabs[feature].values
})

print("\n26. Comparison of scaling techniques:")
print(scaling_comparison.head())

# ============================================================
# 27. Label Encoding
# ============================================================

label_df = df.copy()

label_df["Gender"] = label_df["Gender"].fillna(
    label_df["Gender"].mode()[0]
)

label_encoder = LabelEncoder()

label_df["Gender_Label"] = label_encoder.fit_transform(
    label_df["Gender"]
)

print("\n27. Label Encoding:")
print(label_df[["Gender", "Gender_Label"]].head())

# ============================================================
# 28. One-Hot Encoding
# ============================================================

one_hot_df = pd.get_dummies(
    df,
    columns=["City"],
    dtype=int
)

print("\n28. One-Hot Encoding:")
print(one_hot_df.head())

# ============================================================
# 29. Binary Encoding on categorical data
# ============================================================

# Binary encoding can be performed without an external package.
# The category is first converted to an integer and then represented
# using binary digits.

binary_df = df.copy()

binary_df["City"] = binary_df["City"].fillna(
    binary_df["City"].mode()[0]
)

city_categories = {
    category: index
    for index, category in enumerate(
        binary_df["City"].unique()
    )
}

binary_df["City_Code"] = binary_df["City"].map(
    city_categories
)

max_code = int(binary_df["City_Code"].max())
bits = max(1, len(bin(max_code)) - 2)

for bit in range(bits):
    binary_df[f"City_Bit_{bit}"] = (
        binary_df["City_Code"]
        .apply(lambda value: (int(value) >> bit) & 1)
    )

print("\n29. Binary Encoding:")
print(binary_df.head())

# ============================================================
# 30. Create a new feature by combining two existing columns
# ============================================================

feature_df = df.copy()

feature_df["Age"] = feature_df["Age"].fillna(
    feature_df["Age"].median()
)

feature_df["Marks"] = feature_df["Marks"].fillna(
    feature_df["Marks"].median()
)

feature_df["Age_Marks"] = (
    feature_df["Age"] * feature_df["Marks"]
)

print("\n30. New combined feature Age_Marks:")
print(feature_df[["Age", "Marks", "Age_Marks"]].head())

# ============================================================
# 31. Extract year, month and day from a date column
# ============================================================

date_df = df.copy()

date_df["Date"] = pd.to_datetime(date_df["Date"])

date_df["Year"] = date_df["Date"].dt.year
date_df["Month"] = date_df["Date"].dt.month
date_df["Day"] = date_df["Date"].dt.day

print("\n31. Extracted Year, Month and Day:")
print(
    date_df[
        ["Date", "Year", "Month", "Day"]
    ].head()
)

# ============================================================
# 32. Create a new feature using mathematical transformations
# ============================================================

math_df = outlier_df.copy()

math_df["Marks_Squared"] = math_df["Marks"] ** 2
math_df["Age_Log"] = np.log1p(math_df["Age"])

print("\n32. Mathematical transformation:")
print(
    math_df[
        ["Age", "Marks", "Marks_Squared", "Age_Log"]
    ].head()
)

# ============================================================
# 33. Apply Log Transformation to skewed data
# ============================================================

log_df = outlier_df.copy()

log_df["Income_Log"] = np.log1p(
    log_df["Income"]
)

print("\n33. Log-transformed Income:")
print(
    log_df[
        ["Income", "Income_Log"]
    ].head()
)

# Visualize original vs log-transformed data
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(
    log_df["Income"],
    bins=10
)
axes[0].set_title("Original Income")

axes[1].hist(
    log_df["Income_Log"],
    bins=10
)
axes[1].set_title("Log Transformed Income")

plt.tight_layout()
plt.show()

# ============================================================
# 34. Feature Selection using correlation analysis
# ============================================================

correlation_matrix = outlier_df[numeric_columns].corr()

print("\n34. Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Analysis")
plt.tight_layout()
plt.show()

# Select features strongly correlated with Marks
target = "Marks"

correlations = (
    outlier_df[numeric_columns]
    .corr()[target]
    .abs()
    .sort_values(ascending=False)
)

print("\nFeatures ranked by correlation with Marks:")
print(correlations)

# ============================================================
# 35. Create final preprocessed dataset ready for Machine Learning
# ============================================================

final_df = df.copy()

# Convert date
final_df["Date"] = pd.to_datetime(final_df["Date"])

# Extract date features
final_df["Year"] = final_df["Date"].dt.year
final_df["Month"] = final_df["Date"].dt.month
final_df["Day"] = final_df["Date"].dt.day

# Remove original date column
final_df = final_df.drop(columns=["Date"])

# Fill numerical missing values using median
final_numeric_columns = final_df.select_dtypes(
    include=np.number
).columns

for column in final_numeric_columns:
    final_df[column] = final_df[column].fillna(
        final_df[column].median()
    )

# Fill categorical missing values using mode
final_categorical_columns = final_df.select_dtypes(
    exclude=np.number
).columns

for column in final_categorical_columns:
    if not final_df[column].mode().empty:
        final_df[column] = final_df[column].fillna(
            final_df[column].mode()[0]
        )

# One-hot encode categorical variables
final_df = pd.get_dummies(
    final_df,
    columns=final_categorical_columns,
    dtype=int
)

# Scale numerical features
final_numeric_columns = final_df.select_dtypes(
    include=np.number
).columns

final_scaler = StandardScaler()

final_df[final_numeric_columns] = (
    final_scaler.fit_transform(
        final_df[final_numeric_columns]
    )
)

# Save final preprocessed dataset
FINAL_FILE = "labsheet_02_final_preprocessed.csv"

final_df.to_csv(
    FINAL_FILE,
    index=False
)

print("\n35. Final preprocessed dataset:")
print(final_df.head())

print("\nFinal dataset shape:", final_df.shape)
print("Missing values remaining:")
print(final_df.isnull().sum())

print(
    f"\nFinal preprocessed dataset saved as: {FINAL_FILE}"
)
