 # Common imports used throughout the notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes, load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (7, 5)


# ## 1. Load a Regression Dataset using Pandas

diabetes_data = load_diabetes(as_frame=True)
df = diabetes_data.frame.copy()

print("Dataset shape:", df.shape)
print("\n" + diabetes_data.DESCR.split('\n\n')[1])   # short description
df.head()


# ## 2. Display First and Last Five Records

print("First 5 records:")
display(df.head())

print("\nLast 5 records:")
display(df.tail())


# ## 3. Explore Dataset Information and Descriptive Statistics

print("Dataset Info:")
df.info()

print("\nDescriptive Statistics:")
df.describe()


# ## 4. Identify Input (Independent) and Output (Dependent) Variables
# - **Independent variables (X):** `age, sex, bmi, bp, s1, s2, s3, s4, s5, s6`
#   (age, sex, body-mass index, average blood pressure, and six blood-serum
#   measurements)
# - **Dependent / target variable (y):** `target` — a quantitative measure of
#   disease progression one year after baseline
# 
# For the *Simple* Linear Regression experiments we will use a single, highly
# correlated feature: `bmi` (body mass index).

independent_vars = df.columns.drop('target').tolist()
dependent_var = 'target'

print("Independent variables:", independent_vars)
print("Dependent variable  :", dependent_var)

# Correlation with target to justify choice of feature for Simple Linear Regression
print("\nCorrelation of each feature with target:")
print(df.corr(numeric_only=True)[dependent_var].sort_values(ascending=False))


# ## 5. Split the Dataset into Training and Testing Sets

X_simple = df[['bmi']]          # single feature for Simple Linear Regression
X_multi  = df[independent_vars]     # all features for Multiple Linear Regression
y = df[dependent_var]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.2, random_state=42)

print("Simple  -> Train:", X_train_s.shape, " Test:", X_test_s.shape)
print("Multiple-> Train:", X_train_m.shape, " Test:", X_test_m.shape)


# ## 6 & 7. Implement and Train a Simple Linear Regression Model

simple_lr = LinearRegression()
simple_lr.fit(X_train_s, y_train_s)
print("Simple Linear Regression model trained.")


# ## 8. Predict Output Values using the Trained Model

y_pred_s = simple_lr.predict(X_test_s)
print("First 10 predictions:", np.round(y_pred_s[:10], 3))


# ## 9. Visualize the Linear Regression Line

plt.figure()
plt.scatter(X_test_s, y_test_s, s=10, alpha=0.4, label='Actual data')
# sort for a clean line
order = np.argsort(X_test_s['bmi'].values)
plt.plot(X_test_s['bmi'].values[order], y_pred_s[order], color='red', linewidth=2, label='Regression line')
plt.xlabel('BMI (bmi)')
plt.ylabel('Disease Progression Score')
plt.title('Simple Linear Regression: bmi vs target')
plt.legend()
plt.show()


# ## 10. Compare Actual and Predicted Values

comparison_df = pd.DataFrame({'Actual': y_test_s.values, 'Predicted': y_pred_s})
comparison_df['Error'] = comparison_df['Actual'] - comparison_df['Predicted']
comparison_df.head(10)


# ## 11. Display the Regression Coefficient and Intercept

print("Coefficient (slope):", simple_lr.coef_[0])
print("Intercept           :", simple_lr.intercept_)
print(f"\nEquation: target = {simple_lr.coef_[0]:.4f} * bmi + {simple_lr.intercept_:.4f}")


# ## 12. Predict Output for New User-Defined Input Values

# Note: this dataset's features are already mean-centered / scaled by scikit-learn,
# so realistic user-defined BMI inputs fall roughly in the range [-0.09, 0.17].
new_bmi_values = pd.DataFrame({'bmi': [-0.05, 0.0, 0.08]})   # user-defined inputs
new_predictions = simple_lr.predict(new_bmi_values)

for inc, pred in zip(new_bmi_values['bmi'], new_predictions):
    print(f"bmi = {inc} -> Predicted target = {pred:.3f}")


# ## 13 & 14. Implement and Train a Multiple Linear Regression Model

multi_lr = LinearRegression()
multi_lr.fit(X_train_m, y_train_m)
print("Multiple Linear Regression model trained on features:", independent_vars)


# ## 15. Predict Output Values using the Testing Dataset

y_pred_m = multi_lr.predict(X_test_m)
print("First 10 predictions:", np.round(y_pred_m[:10], 3))


# ## 16. Compare Actual and Predicted Values Graphically

plt.figure()
plt.scatter(y_test_m, y_pred_m, s=10, alpha=0.4)
lims = [min(y_test_m.min(), y_pred_m.min()), max(y_test_m.max(), y_pred_m.max())]
plt.plot(lims, lims, color='red', linewidth=2, label='Ideal (Actual = Predicted)')
plt.xlabel('Actual target')
plt.ylabel('Predicted target')
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.legend()
plt.show()


# ## 17. Analyze the Effect of Each Independent Variable on Prediction

coef_df = pd.DataFrame({
    'Feature': independent_vars,
    'Coefficient': multi_lr.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

display(coef_df)

plt.figure()
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='steelblue')
plt.xlabel('Coefficient Value')
plt.title('Effect of Each Independent Variable on Predicted Disease Progression')
plt.gca().invert_yaxis()
plt.show()

print("Interpretation: a larger absolute coefficient means that feature has a")
print("stronger effect on the predicted target, holding other features constant.")
print("The sign shows the direction of the effect (positive = increases target).")


# ## 18. Implement Polynomial Regression of Degree 2

poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train_s)
X_test_poly2  = poly2.transform(X_test_s)

poly_lr2 = LinearRegression()
poly_lr2.fit(X_train_poly2, y_train_s)
y_pred_poly2 = poly_lr2.predict(X_test_poly2)

print("Degree-2 Polynomial Regression trained.")
print("First 10 predictions:", np.round(y_pred_poly2[:10], 3))


# ## 19. Implement Polynomial Regression of Degree 3

poly3 = PolynomialFeatures(degree=3)
X_train_poly3 = poly3.fit_transform(X_train_s)
X_test_poly3  = poly3.transform(X_test_s)

poly_lr3 = LinearRegression()
poly_lr3.fit(X_train_poly3, y_train_s)
y_pred_poly3 = poly_lr3.predict(X_test_poly3)

print("Degree-3 Polynomial Regression trained.")
print("First 10 predictions:", np.round(y_pred_poly3[:10], 3))


# ## 20. Compare Linear Regression and Polynomial Regression Models

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

comparison_table = pd.DataFrame({
    'Model': ['Simple Linear (deg 1)', 'Polynomial (deg 2)', 'Polynomial (deg 3)'],
    'R2 Score': [r2(y_test_s, y_pred_s), r2(y_test_s, y_pred_poly2), r2(y_test_s, y_pred_poly3)]
})
comparison_table


# ## 21. Visualize Polynomial Regression Curves

x_range = np.linspace(X_simple['bmi'].min(), X_simple['bmi'].max(), 300).reshape(-1, 1)

y_line1 = simple_lr.predict(x_range)
y_line2 = poly_lr2.predict(poly2.transform(x_range))
y_line3 = poly_lr3.predict(poly3.transform(x_range))

plt.figure()
plt.scatter(X_test_s, y_test_s, s=8, alpha=0.3, label='Actual data', color='gray')
plt.plot(x_range, y_line1, label='Linear (deg 1)', linewidth=2)
plt.plot(x_range, y_line2, label='Polynomial (deg 2)', linewidth=2)
plt.plot(x_range, y_line3, label='Polynomial (deg 3)', linewidth=2)
plt.xlabel('BMI (bmi)')
plt.ylabel('Disease Progression Score')
plt.title('Linear vs Polynomial Regression Curves')
plt.legend()
plt.show()


# ## 22. Predict Output Values using the Polynomial Regression Model

new_vals_poly = poly2.transform(new_bmi_values)
poly_new_pred = poly_lr2.predict(new_vals_poly)

for inc, pred in zip(new_bmi_values['bmi'], poly_new_pred):
    print(f"bmi = {inc} -> Predicted target (Degree-2 Poly) = {pred:.3f}")


# ## 23. Compare Prediction Accuracy for Different Polynomial Degrees

degree_results = []
for d in range(1, 6):
    poly = PolynomialFeatures(degree=d)
    Xtr = poly.fit_transform(X_train_s)
    Xte = poly.transform(X_test_s)
    model = LinearRegression().fit(Xtr, y_train_s)
    pred = model.predict(Xte)
    degree_results.append({
        'Degree': d,
        'MAE': mean_absolute_error(y_test_s, pred),
        'MSE': mean_squared_error(y_test_s, pred),
        'R2': r2_score(y_test_s, pred)
    })

degree_df = pd.DataFrame(degree_results)
display(degree_df)

plt.figure()
plt.plot(degree_df['Degree'], degree_df['R2'], marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('R2 Score')
plt.title('R2 Score vs Polynomial Degree')
plt.show()


# # Regression Model Evaluation

# ## 24–27. Calculate MAE, MSE, RMSE and R² Score

def evaluate_model(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}\n")
    return {'Model': name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

results = []
results.append(evaluate_model(y_test_s, y_pred_s, "Simple Linear Regression"))
results.append(evaluate_model(y_test_m, y_pred_m, "Multiple Linear Regression"))
results.append(evaluate_model(y_test_s, y_pred_poly2, "Polynomial Regression (deg 2)"))
results.append(evaluate_model(y_test_s, y_pred_poly3, "Polynomial Regression (deg 3)"))


# ## 28. Compare the Performance of Linear and Polynomial Regression using Evaluation Metrics

results_df = pd.DataFrame(results)
results_df


# ## 29. Interpret the Meaning of MSE and R² Values
# 
# - **MSE (Mean Squared Error)** measures the average squared difference between
#   actual and predicted values. It penalizes larger errors more heavily
#   (because of squaring). A **lower MSE** means the model's predictions are,
#   on average, closer to the actual values. Its units are the square of the
#   target's units, which makes it harder to interpret directly — **RMSE**
#   (its square root) is often preferred because it is in the same units as
#   the target.
# - **R² (R-squared / Coefficient of Determination)** measures the proportion
#   of variance in the dependent variable that is explained by the independent
#   variable(s). It ranges (typically) from 0 to 1 (it can be negative for a
#   very poor model): an R² of **0.65**, for example, means the model explains
#   65% of the variability in disease progression, while the remaining 35% is due to
#   factors not captured by the model or to inherent noise.
# 
# Together, a good regression model should have a **low MSE/RMSE** and a
# **high R²**.

# ## 30. Visualize Prediction Errors using Scatter Plots

errors_simple = y_test_s.values - y_pred_s
errors_multi = y_test_m.values - y_pred_m

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_pred_s, errors_simple, s=10, alpha=0.4)
axes[0].axhline(0, color='red', linewidth=2)
axes[0].set_xlabel('Predicted Values')
axes[0].set_ylabel('Residual (Actual - Predicted)')
axes[0].set_title('Residuals: Simple Linear Regression')

axes[1].scatter(y_pred_m, errors_multi, s=10, alpha=0.4, color='green')
axes[1].axhline(0, color='red', linewidth=2)
axes[1].set_xlabel('Predicted Values')
axes[1].set_ylabel('Residual (Actual - Predicted)')
axes[1].set_title('Residuals: Multiple Linear Regression')

plt.tight_layout()
plt.show()


# # Model Improvement and Analysis

# ## 31. Train the Regression Model using Standardized Features

scaler = StandardScaler()
X_train_m_scaled = scaler.fit_transform(X_train_m)
X_test_m_scaled  = scaler.transform(X_test_m)

multi_lr_scaled = LinearRegression()
multi_lr_scaled.fit(X_train_m_scaled, y_train_m)
y_pred_m_scaled = multi_lr_scaled.predict(X_test_m_scaled)

print("Multiple Linear Regression trained on standardized features.")


# ## 32. Compare Model Performance Before and After Feature Scaling

before_scaling = evaluate_model(y_test_m, y_pred_m, "Multiple LR (Before Scaling)")
after_scaling  = evaluate_model(y_test_m, y_pred_m_scaled, "Multiple LR (After Scaling)")

scaling_compare_df = pd.DataFrame([before_scaling, after_scaling])
display(scaling_compare_df)

print("Note: Ordinary Least Squares Linear Regression is scale-invariant in terms")
print("of predictive performance (R2/MSE stay essentially the same) because scaling")
print("only rescales the coefficients. Feature scaling mainly helps gradient-based")
print("or regularized models (e.g. Ridge/Lasso/SGD) converge faster and fairly")
print("penalize coefficients across differently-scaled features.")


# ## 33. Perform Regression using Another Real-World Dataset (Linnerud dataset)
# 
# The **Linnerud** dataset records physiological and exercise measurements
# from 20 members of a fitness club: three exercise variables (`Chins`,
# `Situps`, `Jumps`) are used here to predict the physiological variable
# `Weight`.

linnerud = load_linnerud(as_frame=True)
X_l = linnerud.data          # Chins, Situps, Jumps
y_l = linnerud.target['Weight']

df_linnerud = pd.concat([X_l, y_l], axis=1)
print("Linnerud dataset shape:", df_linnerud.shape)
display(df_linnerud)

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l, y_l, test_size=0.25, random_state=42)

linnerud_lr = LinearRegression()
linnerud_lr.fit(X_train_l, y_train_l)
y_pred_l = linnerud_lr.predict(X_test_l)

evaluate_model(y_test_l, y_pred_l, "Linear Regression on Linnerud Dataset (predicting Weight)")


# ## 34. Save the Trained Regression Model using the Joblib Library

import os
os.makedirs('models', exist_ok=True)

joblib.dump(multi_lr, 'models/multiple_linear_regression_model.joblib')
joblib.dump(scaler, 'models/feature_scaler.joblib')

print("Model saved to: models/multiple_linear_regression_model.joblib")


# ## 35. Load the Saved Model and Use it to Predict New Data

loaded_model = joblib.load('models/multiple_linear_regression_model.joblib')

# Build a small sample of new, unseen input data using the same columns
sample_new_data = X_test_m.iloc[:5]
loaded_predictions = loaded_model.predict(sample_new_data)

result_df = pd.DataFrame({
    'Actual': y_test_m.iloc[:5].values,
    'Predicted (Loaded Model)': loaded_predictions
})
result_df
