# logisticregression model with iris dataset. model training is done with only sepal length.
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np
dataset = load_iris()
df = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
df = df.drop(columns=df.columns[1:4])
mapped_array = np.array(dataset['target_names'])[dataset['target']]
X_train, X_test, y_train, y_test = train_test_split(df, mapped_array, test_size=0.2, random_state=42)
model = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
model.fit(X_train, y_train)
custom_input = pd.DataFrame([[5.9]], columns=['sepal length (cm)'])
y_pred = model.predict(custom_input)
print(y_pred)
