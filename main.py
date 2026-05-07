import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE


df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)


# Remove customerID column
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

print("\nMISSING VALUES")
print(df.isnull().sum())


# Save encoders
encoders = {}

# Find categorical columns
object_columns = df.select_dtypes(include=["object"]).columns

# Encode columns
for column in object_columns:

    label_encoder = LabelEncoder()

    df[column] = label_encoder.fit_transform(df[column])

    encoders[column] = label_encoder

# Save encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("\nENCODING COMPLETED")


X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nTARGET VALUE COUNTS")
print(y.value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING DATA SHAPE")
print(X_train.shape)

print("\nTEST DATA SHAPE")
print(X_test.shape)


smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAFTER SMOTE")
print(y_train_smote.value_counts())


models = {

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
}

# Store trained models
trained_models = {}

# Cross-validation scores
cv_scores = {}


for model_name, model in models.items():

    print("\n" + "="*60)
    print(f"TRAINING: {model_name}")
    print("="*60)

    # Cross Validation
    scores = cross_val_score(
        model,
        X_train_smote,
        y_train_smote,
        cv=5,
        scoring="accuracy"
    )

    cv_scores[model_name] = scores

    print(f"\nCross Validation Scores:")
    print(scores)

    print(f"\nAverage CV Accuracy:")
    print(np.mean(scores))

    # Train model
    model.fit(X_train_smote, y_train_smote)

    # Save model
    trained_models[model_name] = model

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy:")
    print(accuracy)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


best_model_name = max(
    cv_scores,
    key=lambda x: np.mean(cv_scores[x])
)

best_model = trained_models[best_model_name]

print("\n" + "="*60)
print(f"BEST MODEL: {best_model_name}")
print("="*60)



with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nBEST MODEL SAVED SUCCESSFULLY")




if best_model_name != "Decision Tree":

    feature_importance = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=feature_importance.values,
        y=feature_importance.index
    )

    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")

    plt.tight_layout()

    plt.show()




print("\nPROJECT COMPLETED SUCCESSFULLY")