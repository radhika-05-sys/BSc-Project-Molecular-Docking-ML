import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, fbeta_score

csv_path = "LIGANDS.csv"
df = pd.read_csv(csv_path)

column_map = {
    "Binding Affinity": "Binding_Affinity",
    "Molecular_Weight (g/mol)": "Molecular_Weight",
    "TPSA  (Ų)": "TPSA"
}
df = df.rename(columns=column_map)

best_idx = df["Binding_Affinity"].idxmin()
best_row = df.loc[best_idx]
print("\nBest binding affinity in dataset:")
print(
    f"Ligand: {best_row['Ligand']} | Pose: {best_row['Pose']} | Binding Affinity: {best_row['Binding_Affinity']}"
)
features = [
    "Molecular_Weight",
    "HBAC",
    "HBDC",
    "Lipinski_Violations",
    "XlogP",
    "TPSA"
]
target = "Binding_Affinity"
X = df[features]
y = df[target]
print("\nTarget variable:")
print(target)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("\nNumber of training samples:", X_train.shape[0])
print("Number of testing samples:", X_test.shape[0])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
validation_score = cv_scores.mean()
median_affinity = y_test.median()
y_test_binary = (y_test < median_affinity).astype(int)
y_pred_binary = (y_pred < median_affinity).astype(int)
f2_score_value = fbeta_score(y_test_binary, y_pred_binary, beta=2, zero_division=0)

print("\nModel Evaluation Results")
print("R2 Score:", round(r2, 3))
print("Validation Score:", round(validation_score, 3))
print("F2 Score:", round(f2_score_value, 3))
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, color="blue", edgecolor="black")
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())
plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")
plt.xlabel("Actual Binding Affinity (kcal/mol)")
plt.ylabel("Predicted Binding Affinity (kcal/mol)")
plt.title("Actual vs Predicted Binding Affinity")
plt.show()

coefficient_table = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})
coefficient_table["Absolute_Coefficient"] = coefficient_table["Coefficient"].abs()
coefficient_table = coefficient_table.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)
print("\nLinear Regression coefficients:")
print(coefficient_table)

results = pd.DataFrame({
    "Actual_Binding_Affinity": y_test.values,
    "Predicted_Binding_Affinity": y_pred
})
results.to_csv("inha_ligand_prediction.csv", index=False)
print("\nPrediction results saved as: inha_ligand_prediction.csv")