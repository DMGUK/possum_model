import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('possum.csv')
print(data.to_string())

X1 = data['hdlngth'].values
X2 = data['skullw'].values
Y = data['totlngth'].values

mean_x1 = np.mean(X1)
mean_x2 = np.mean(X2)
mean_y = np.mean(Y)

m1 = len(X1)
m2 = len(X2)

numer1 = 0
numer2 = 0
denom1 = 0
denom2 = 0

for i in range(m1):
    numer1 += (X1[i] - mean_x1) * (Y[i] - mean_y)
    denom1 += (X1[i] - mean_x1) ** 2

for i in range(m2):
    numer2 += (X2[i] - mean_x2) * (Y[i] - mean_y)
    denom2 += (X2[i] - mean_x2) ** 2

m1 = numer1 / denom1
m2 = numer2 / denom2

c = mean_y - (m1 * mean_x1) - (m2 * mean_x2)

print(f'm1 = {m1}\nm2 = {m2}\nc = {c}')

max_x1 = np.max(X1)
min_x1 = np.min(X1)

max_x2 = np.max(X2)
min_x2 = np.min(X2)

x1, x2 = np.meshgrid(np.linspace(min_x1, max_x1, 100), np.linspace(min_x2, max_x2, 100))

y = c + m1 * x1 + m2 * x2

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

colors = np.array(['red' if x2 < np.mean(X2) else 'blue' for x2 in X2])
ax.scatter(X1, X2, Y, color=colors, label='Data Points')
ax.plot_surface(x1, x2, y, alpha=0.7, color='green', label='Regression Surface')
ax.set_xlabel('Head length, in mm')
ax.set_ylabel('Skull width, in mm')
ax.set_zlabel('Total length, in cm')
dummy = plt.Line2D([0], [0], color='green', label='Regression Surface')
dummy2 = plt.Line2D([0], [0], color='red', label='Data Points from x1')
dummy3 = plt.Line2D([0], [0], color='blue', label='Data Points from x2')
plt.legend(handles=[dummy, dummy2, dummy3])
plt.show()

ss_t = 0
ss_r = 0

for i in range(int(len(X1))):
    y_pred = c + m1 * X1[i] + m2 * X2[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2

r2 = 1 - (ss_r / ss_t)

print(f'R-squared: {r2}')
