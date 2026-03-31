# import pandas as pd
# import numpy as np 
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score
# import joblib 

# # Load Dataset
# train_df = pd.read_csv("Training.csv")
# test_df = pd.read_csv("Testing.csv")

# # Seperate Features and Target 
# X_train = train_df.drop("prognosis", axis=1)
# y_train = train_df["prognosis"]

# X_test = test_df.drop("prognosis", axis=1)
# y_test = test_df["prognosis"]

# # Encode Disease Labels 
# le = LabelEncoder()
# y_train_encoded = le.fit_transform(y_train)
# y_test_encoded = le.transform(y_test)

# # Create Model 
# model = MultinomialNB()

# # Train Model 
# model.fit(X_train, y_train_encoded)

# # Test Accuracy 
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test_encoded, y_pred)

# print("Model Accuracy:", accuracy)

# # Save Model and Encoder 
# joblib.dump(model, "trained_model.pkl")
# joblib.dump(le, "label_encoder.pkl")

# # Save Column Order (VVI)
# joblib.dump(X_train.columns.tolist(), "model_columns.pkl")

# print("Model training completed successfully!")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pickle as pickle

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("Training.csv")

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# =========================
# 2. Define Stratified K-Fold
# =========================
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4222222)

# =========================
# 3. Define Model
# =========================
model = MultinomialNB()

# =========================
# 4. Hyperparameter Grid
# =========================
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
    'fit_prior': [True, False]
}

# =========================
# 5. Grid Search with Stratified K-Fold
# =========================
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X, y)

# =========================
# 6. Best Model
# =========================
best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# =========================
# 7. Cross-Validated Predictions (NO DATA LEAKAGE)
# =========================
y_pred = cross_val_predict(best_model, X, y, cv=kfold)

# =========================
# 8. Evaluation Metrics
# =========================
print("\nEvaluation Metrics (Stratified K-Fold)")

print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred, average='weighted'))
print("Recall   :", recall_score(y, y_pred, average='weighted'))
print("F1 Score :", f1_score(y, y_pred, average='weighted'))

# =========================
# 9. Confusion Matrix
# =========================
# cm = confusion_matrix(y, y_pred)

# plt.figure(figsize=(16,12))
# sn.heatmap(cm, annot=True, fmt='d', cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix For PREDICO")

# # Save as JPG (high quality)
# # plt.savefig("confusion_matrix.jpg", format="jpg", dpi=300, bbox_inches='tight')
# plt.show()

# =========================
# 9. Save Best Model
# =========================
pickle.dump(best_model, open("multinomial_kfold_model.pkl", "wb"))

print("\nModel saved successfully!")