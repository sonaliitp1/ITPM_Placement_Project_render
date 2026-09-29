
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------- 1. Load and split ----------
df = pd.read_csv("data/placement.csv")

FEATURES = ["IQ", "CGPA", "10th_Marks", "12th_Marks", "Communication_Skills"]
X = df[FEATURES]
y = df["Placed"]

# 80% train, 20% test. stratify keeps the placed/not-placed ratio same in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")


# ---------- 2. Baseline model ----------
# Random Forest = many decision trees, each trained on a random sample of data
# and random subset of features. Final answer = majority vote of all trees.
baseline = RandomForestClassifier(random_state=42)
baseline.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
print(f"Baseline Random Forest accuracy: {baseline_acc:.4f}")


# ---------- 3. Hyperparameter tuning ----------
# GridSearchCV tries every combination below and picks the best using
# 5-fold cross-validation on the TRAINING data only (test data stays untouched).
param_grid = {
    "n_estimators":      [100, 200],      # number of trees
    "max_depth":         [None, 10],    # how deep each tree can grow
    "min_samples_split": [2, 5],           # min rows needed to split a node
    "min_samples_leaf":  [1, 2],            # min rows in a leaf node
    "max_features":      ["sqrt",None]    # features considered at each split
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,       # use all CPU cores
    
)
grid.fit(X_train, y_train)

print("\nBest hyperparameters:", grid.best_params_)
print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")

# ---------- 4. Evaluate best model ----------
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\nTuned Random Forest TEST accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"(Improvement over baseline: {accuracy_score(y_test, y_pred) - baseline_acc:+.4f})")

print("\nConfusion Matrix (rows = actual, cols = predicted):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

# 5-fold CV on the whole dataset gives a more reliable estimate than one test split
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"5-fold CV accuracy on full data: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ---------- 5. Feature importance + save ----------
importance = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importance:")
print(importance.round(3))

pickle.dump(best_model, open("placement_model.pkl", "wb"))
print("\nModel saved as placement_model.pkl")


