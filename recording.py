import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Requirements/advertising.csv')

x = np.array(data.drop(['Sales'], axis=1))
y = np.array(data['Sales'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

scalar = StandardScaler()
x_norm = scalar.fit_transform(x_train)
print(f'Peek to Peek range by column in Raw        x: {np.ptp(x_train, axis=0)}')
print(f'Peek to Peek range by column in Normalized x: {np.ptp(x_norm, axis=0)}\n')

model = LinearRegression()
model.fit(x_norm, y_train)

y_predict = model.predict(x_norm)

score = model.score(x_norm, y_train)

print(f'Score = {score: .2f}')

features = np.array([[230, 38, 90]])
print(f'Predict: {model.predict(features)[0]: .2f}')
