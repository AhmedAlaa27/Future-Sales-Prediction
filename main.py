import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('Requirements/advertising.csv')

correlation = data.corr()
print(correlation['Sales'].sort_values(ascending=False))

x = np.array(data.drop(['Sales'], axis=1))
y = np.array(data['Sales'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

scalar = StandardScaler()
x_norm = scalar.fit_transform(x_train)
print(f"Peak to Peak range by column in Raw        X: {np.ptp(x_train, axis=0)}")
print(f"Peak to Peak range by column in Normalized X: {np.ptp(x_norm, axis=0)}\n")

model = LinearRegression()
model.fit(x_norm, y_train)

y_predict = model.predict(x_norm)

print(f'Score: {model.score(x_norm, y_train): .2f}')

# Features --> [ TV, Radio, Newspaper ]
features = np.array([[230, 37.8, 89]])
print(f'Predict: {model.predict(features)[0]: .2f}')

fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
heading = np.array(['TV', 'Radio', 'Newspaper'])
for i in range(len(ax)):
    ax[i].scatter(x_train[:, i], y_train, label='target')
    ax[i].set_xlabel(heading[i])
    ax[i].scatter(x_train[:, i], y_predict, color='orange', label='predict')
ax[0].set_ylabel("Price")
ax[0].legend()
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
