
# Titanic Survival Prediction using Naive Bayes

## Overview
This project focuses on predicting whether a passenger survived or not on the Titanic using machine learning. The dataset used for this project includes various features like age, gender, passenger class, and other attributes. The model is built using the Naive Bayes algorithm.

## Table of Contents
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Acknowledgments](#acknowledgments)

## Data Preprocessing
The Titanic dataset is cleaned and preprocessed by:
- Dropping irrelevant columns (`PassengerId`, `Name`, `Ticket`, etc.)
- Filling missing values, such as missing ages, with the mean age.
- Encoding categorical variables (e.g., `Sex`) using `LabelEncoder`.

## Model Training
The model is trained using the `GaussianNB` (Naive Bayes) algorithm from the `sklearn` library. It predicts survival (`Survived` column) based on the remaining features in the dataset.

## Model Evaluation
After training the model, it is evaluated on a test set, and the accuracy score is printed.

### Example Code:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the Titanic dataset
df = pd.read_csv("titanic.csv")
df.drop(['PassengerId', 'Name', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'SibSp'], axis='columns', inplace=True)
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# Handle missing values by filling missing Age with the mean age
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Prepare features and target variable
target = df['Survived']
inputs = df.drop('Survived', axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)

# Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate the model accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Predict on the first 10 test samples
predictions = model.predict(X_test[:10])
print(f"Predictions: {predictions}")
```

## Requirements
To run the project, you will need the following Python libraries:
- pandas
- scikit-learn

You can install them using:
```bash
pip install pandas scikit-learn
```

## How to Run
1. Clone this repository:
```bash
git clone https://github.com/Bushra-Butt-17/titanic-survival-prediction.git
```
2. Change directory into the project folder:
```bash
cd titanic-survival-prediction
```
3. Install the required libraries:
```bash
pip install -r requirements.txt
```
4. Run the script to train and evaluate the model:
```bash
python titanic_prediction.py
```

## Acknowledgments
- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
```
