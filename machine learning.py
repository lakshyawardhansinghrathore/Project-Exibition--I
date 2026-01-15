import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("cleaned_diseases.csv")

# Split dataset into features (X) and target (y)
y = df["diseases"]
X = df.drop("diseases", axis=1)

# Remove classes with only one instance
class_counts = y.value_counts()
single_instance_classes = class_counts[class_counts == 1].index
if len(single_instance_classes) > 0:
    df = df[~df["diseases"].isin(single_instance_classes)]
    y = df["diseases"]
    X = df.drop("diseases", axis=1)

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Ensure all columns in X are numeric and fill missing values
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X_dense = X.values

# Set the number of components for dimensionality reduction
n_components = 50

# Apply SVD and NMF
svd = TruncatedSVD(n_components=n_components, random_state=100)
X_reduced_svd = svd.fit_transform(X_dense)
nmf = NMF(n_components=n_components, init='random', random_state=100)
X_reduced_nmf = nmf.fit_transform(X_dense)

# Function to classify using cosine similarity
def cosine_similarity_classifier(X_train, y_train, X_test):
    cos_sim_matrix = cosine_similarity(X_test, X_train)
    predictions = [y_train[idx] for idx in cos_sim_matrix.argmax(axis=1)]
    return predictions

# Dictionary of similarity-based classifiers
similarity_classifiers = {
    "Cosine Similarity": cosine_similarity_classifier
}

# Dictionary of sklearn models
models = {
    'XGBoost': XGBClassifier(random_state=100, eval_metric='mlogloss'),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, random_state=100)
}

# Results dictionary to store model performance metrics
results = {
    "Model": [],
    "Transformation": [],
    "Validation Accuracy": [],
    "Test Accuracy": []
}

# Loop through transformations: SVD and NMF
for transform_name, X_reduced in [("SVD", X_reduced_svd), ("NMF", X_reduced_nmf)]:
    # Split data into training and test sets
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
        X_reduced, y, test_size=0.1, random_state=100, stratify=y)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=100, stratify=y_train_full)

    # Custom similarity classifier (Cosine)
    for model_name, classifier in similarity_classifiers.items():
        print(f"Evaluating {model_name} using {transform_name}")
        y_val_pred = classifier(X_train, y_train, X_val)
        y_test_pred = classifier(X_train, y_train, X_test_final)

        # Calculate accuracy
        val_accuracy = accuracy_score(y_val, y_val_pred)
        test_accuracy = accuracy_score(y_test_final, y_test_pred)

        # Append results
        results["Model"].append(model_name)
        results["Transformation"].append(transform_name)
        results["Validation Accuracy"].append(val_accuracy)
        results["Test Accuracy"].append(test_accuracy)

    # Sklearn models (XGBoost, KNN, SVM)
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name} using {transform_name}")

        # Fit the model
        model.fit(X_train, y_train)

        # Predictions
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test_final)

        # Calculate accuracy
        val_accuracy = accuracy_score(y_val, y_val_pred)
        test_accuracy = accuracy_score(y_test_final, y_test_pred)

        # Append results
        results["Model"].append(model_name)
        results["Transformation"].append(transform_name)
        results["Validation Accuracy"].append(val_accuracy)
        results["Test Accuracy"].append(test_accuracy)

# Create a DataFrame from the results dictionary
results_df = pd.DataFrame(results)

# Display the results
print("Model Performance Summary:")
print(results_df)

# Plot Test Accuracy for each model
plt.figure(figsize=(12, 8))
sns.barplot(x="Model", y="Test Accuracy", hue="Transformation", data=results_df, palette="viridis")
plt.title("Test Accuracy of Each Model with SVD and NMF")
plt.xticks(rotation=45)
plt.show()

# Save the best-performing model and label encoder for future use
best_model = XGBClassifier(random_state=100, eval_metric='mlogloss')
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'best_xgboost_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
