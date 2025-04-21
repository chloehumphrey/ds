# Data Science Assignment 2
# Logistic regression, decision tree, and neural network classifiers

# Bank Marketing Dataset
# https://archive.ics.uci.edu/ml/datasets/bank+marketing

from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# Fetch the Bank Marketing dataset
bank_marketing = fetch_ucirepo(id=222)
# Access feature and target data
X = bank_marketing.data.features
y = bank_marketing.data.targets['y'].map({'yes': 1, 'no': 0})

print('y counts', y.value_counts())

# Preprocess the data ------------------------------------

print('Columns: ', X.columns)
print('Data types: ', X.dtypes)

# Identify categorical and numerical columns
numeric_cols = [
    'age', 'balance', 'campaign', 'duration', 'pdays', 'previous'
]
categorical_cols = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'
]

numeric_transformer = [
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler())
]
categorical_transformer = [
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
]

numeric_pipeline = Pipeline(steps=numeric_transformer)
categorical_pipeline = Pipeline(steps=categorical_transformer)

# Create a preprocessor that applies the numeric and categorical pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ],
    remainder='drop'
)

# Split into testing and training sets -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)

# Train and evaluate models -----------------------------

logreg_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

decision_tree_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

neural_network_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', MLPClassifier(hidden_layer_sizes=(32,16), max_iter=500, random_state=42))
])

models = {
    'Logistic Regression': logreg_pipeline,
    'Decision Tree': decision_tree_pipeline,
    'Neural Network': neural_network_pipeline
}

results = {}
plt.figure(figsize=(15, 5))

for i, (name, model) in enumerate(models.items(), 1):
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }
    
    # Plot confusion matrix
    plt.subplot(1, 3, i)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f'{name}\nConfusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# Print all metrics in a formatted table
print("\nModel Evaluation Metrics:")
print("-" * 80)
print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-score':>10}")
print("-" * 80)
for name, metrics in results.items():
    print(f"{name:<20} {metrics['Accuracy']:>10.4f} {metrics['Precision']:>10.4f} "
          f"{metrics['Recall']:>10.4f} {metrics['F1-score']:>10.4f}")
print("-" * 80)