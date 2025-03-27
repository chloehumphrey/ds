# Data Science Mini Project 2
# Diabetes Health Indicators Dataset

from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

RANDOM_STATE = 42

# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets
  
# metadata 
print(cdc_diabetes_health_indicators.metadata) 
  
# variable information 
print(cdc_diabetes_health_indicators.variables) 

print(X.dtypes)
print(y.dtypes)

# Data Preprocessing ------------------------------------------------------------

y = y['Diabetes_binary']


# 2. Look at feature relavance using Chi Squared

# Display all Chi Squared scores by features
chi2_selector = SelectKBest(chi2, k="all")
X_kbest = chi2_selector.fit_transform(X, y)

chi2_scores = chi2_selector.scores_

chi2_results = pd.DataFrame({
    "Feature": X.columns,
    "Chi2 Score": chi2_scores
})

chi2_results = chi2_results.sort_values(by="Chi2 Score", ascending=False)

print(chi2_results)

# 3. Only keep the 12 most relevant features

selector = SelectKBest(score_func=chi2, k=12)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("Selected Features: ", selected_features)

dropped_features = X.columns[~selector.get_support()]
print("Dropped Features: ", dropped_features)

X = X.drop(dropped_features, axis=1)
print(X.dtypes)

# 1. Standardize all numerical columns in X

# Identify numerical columns
numeric_cols = [
    'BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Income'
]
# Define StandardScaler transformer for numerical columns
numeric_transformer = StandardScaler()
# Create ColumnTransformer that applies StandardScaler only to numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols)
    ],
    remainder='passthrough'  # Keep other columns unchanged
)


# Classifier Pipelines ----------------------------------------------------------

# Logistic Regression pipeline
logistic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=RANDOM_STATE))
])

# Decision Tree pipeline
decision_tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=RANDOM_STATE))
])

# Random Forest pipeline
random_forest_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
])

# MLP Classifier pipeline
mlp_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(hidden_layer_sizes=(10,), max_iter=1500, random_state=RANDOM_STATE))
])

models = {
    'Logistic Regression': logistic_pipeline,
    'Decision Tree': decision_tree_pipeline,

    'Shallow NN': mlp_pipeline
}

# Train & Evaluate Models --------------------------------------------------------

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=RANDOM_STATE)

X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

results = {}
plt.figure(figsize=(15, 5))

for i, (name, model) in enumerate(models.items(), 1):
    # Fit the model
    model.fit(X_train_resampled, y_train_resampled)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)

    # Store the results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

    # Plot the confusion matrix
    plt.subplot(1, 3, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
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
          f"{metrics['Recall']:>10.4f} {metrics['F1']:>10.4f}")
print("-" * 80)