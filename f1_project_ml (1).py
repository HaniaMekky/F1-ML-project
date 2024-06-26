# -*- coding: utf-8 -*-
"""F1 Project ML

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1s0hWNPukFJGYrn5dqNL04ZUmHO13bljq
"""

# in the upcoming steps i will clean the data, modify it and do feature engineering to make the dataset optimum as much as possible and avoid any errors.

#CLEANING

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df_notcleaned = pd.read_csv("Laps.csv")

# Drop unnecessary columns to prevent errors in my coding and some of them are not needed
columns_to_drop = ['IsPersonalBest', 'IsAccurate', 'PitOutTime', 'PitInTime', 'Sector1Time', 'Sector2Time', 'Sector3Time','Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime', 'LapTime', 'Time']
df = df_notcleaned.drop(columns=columns_to_drop)

# Convert 'compound' column to numeric because categorical columns may interupt the model
Compound_mapping = {'MEDIUM': 1, 'HARD': 2, 'SOFT': 3}
df['Compound'] = df['Compound'].map(Compound_mapping)

#Handling missing values
df.dropna(inplace=True)

#Convert categorical variables to numerical
df = pd.get_dummies(df, columns=['Team', 'Driver', 'EventName'], drop_first=True)


# FEATURE ENGINEERING: Difference between the speed at the finish line and start line
df['SPEE_DIFF'] = df['SpeedFL'] - df['SpeedST']

# Print the cleaned Data
print("Cleaned Data:")
print(df.head())



# Save the cleaned data to a new CSV file
df.to_csv("df.csv", index=False)


# Return the cleaned DataFrame
df

#now i will explore the data and use visualization and different graphs

#Exploration and visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a DataFrame
df = pd.read_csv('df.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())

# Describe the dataset's numerical features
print("\nNumerical Features Description:")
print(df.describe())

# Describe the dataset's categorical features
print("\nCategorical Features Description:")
print(df.describe(include=['object']))


# Conduct exploratory data analysis (EDA)
# Histograms of numerical features
df.hist(figsize=(12, 10))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Boxplots of numerical features
plt.figure(figsize=(12, 8))
sns.boxplot(data=df)
plt.title('Boxplots of Numerical Features')
plt.xticks(rotation=45)
plt.show()


# Exploring the  data distribution
plt.figure(figsize=(10, 6))
plt.subplot(121)
df['SpeedI1'].hist(bins=20)
plt.title("SpeedI1 Distribution")
plt.subplot(122)
df['SpeedI2'].hist(bins=20)
plt.title("SpeedI2 Distribution")
plt.show()

# the relationship between SpeedI1 and SpeedI2
plt.scatter(df['SpeedI1'], df['SpeedI2'])
plt.title("SpeedI1 vs. SpeedI2")
plt.show()

# The average points for the speeds
print(df.groupby('SpeedI1')['SpeedI2'].mean())



# i will test different models

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import time

# Load the dataset into a DataFrame
df = pd.read_csv('df.csv')

# Convert all columns to numeric, forcing errors to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Select features and target
X = df[['SpeedI1', 'SpeedI2', 'Compound', 'TyreLife', 'FreshTyre']]
y = df['SpeedFL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




from sklearn.feature_selection import RFE

# Initialize the Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier()

# Initialize RFE
rfe = RFE(estimator=gb_classifier, n_features_to_select=10)

# Fit RFE to the training data
rfe.fit(X_train, y_train)

# Get the selected features
selected_features_rfe = X.columns[rfe.support_]

print("Selected features using RFE:")
print(selected_features_rfe)



# Fit Gradient Boosting classifier to get feature importance
gb_classifier.fit(X_train, y_train)

# Get feature importance scores
feature_importance_gb = gb_classifier.feature_importances_

# Create a DataFrame to visualize feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance_gb
})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Select top features based on importance
selected_features_gb = feature_importance_df.head(10)['Feature'].values

print("Selected features using Gradient Boosting feature importance:")
print(selected_features_gb)

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize models
logistic_regression = LogisticRegression()
random_forest = RandomForestClassifier()
gradient_boosting = GradientBoostingClassifier()
knn = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier()

# Train and evaluate models
models = {
    "Logistic Regression": logistic_regression,
    "Random Forest": random_forest,
    "Gradient Boosting": gradient_boosting,
    "K-Nearest Neighbors": knn,
    "Decision Tree": decision_tree
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Print results
for name, accuracy in results.items():
    print(f"{name} Accuracy: {accuracy:.2f}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('df.csv')

# Convert all columns to numeric, forcing errors to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)



# Define the parameter grid for RandomizedSearchCV
param_distributions = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}

# Initialize the model
rf = RandomForestClassifier()

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Print the best parameters
print(f"Best parameters found: {random_search.best_params_}")

# Evaluate the best model
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Random Forest Accuracy after tuning: {accuracy:.2f}")
print(f"Random Forest Precision after tuning: {precision:.2f}")
print(f"Random Forest Recall after tuning: {recall:.2f}")

# Evaluate using other metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
file='F1LAP'
joblib.dump(best_rf, "best_random_forest_model.pkl")
model=joblib.load(open("best_random_forest_model.pkl",'rb'))