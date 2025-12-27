# logisticregression with breast cancer dataset.
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np
dataset = load_breast_cancer()
df = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
mapped_array = np.array(dataset['target_names'])[dataset['target']]
X_train, X_test, y_train, y_test = train_test_split(df, mapped_array, test_size=0.2, random_state=42)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
custom_input = pd.DataFrame([df.mean().values], columns=df.columns)
y_pred = model.predict(custom_input)
print(y_pred)
