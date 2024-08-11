import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('Requirements/advertising.csv')

x = np.array(data.drop(['Sales'], axis=1))
y = np.array(data['Sales'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

scalar = StandardScaler()
x_norm = scalar.fit_transform(x_train)
print(f"Peak to Peak range by column in Raw        X: {np.ptp(x_train, axis=0)}")
print(f"Peak to Peak range by column in Normalized X: {np.ptp(x_norm, axis=0)}\n")

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(x_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters: w: {w_norm}, b: {b_norm}\n")

y_predict_sgd = sgdr.predict(x_norm)
y_predict = np.dot(x_norm, w_norm) + b_norm
print(f"prediction using np.dot() and sgdr.predict match: {(y_predict == y_predict_sgd).all()}")

print(f"Prediction on training set: {y_predict[:4]}")
print(f"Target values: {y_train[:4]}\n")

score = sgdr.score(x_norm, y_train)
print(f'Score = {score: .2f}')

# # plot predictions and targets vs original features
# fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
# heading = np.array(['TV', 'Radio', 'Newspaper'])
# for i in range(len(ax)):
#     ax[i].scatter(x_train[:, i], y_train, label='target')
#     ax[i].set_xlabel(heading[i])
#     ax[i].scatter(x_train[:, i], y_predict, color='orange', label='predict')
# ax[0].set_ylabel("Price")
# ax[0].legend()
# fig.suptitle("target versus prediction using z-score normalized model")
# plt.show()
