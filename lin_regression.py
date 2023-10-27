import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt

df = pnd.read_csv('possum.csv') 
df.head() 

print(df.shape)

hdl = df['hdlngth'].values.reshape(-1, 1) 
totl = df['totlngth'].values.reshape(-1, 1) 


class Linear_Regression(): 

	def __init__(self, learning_rate, no_of_itr): 
		self.learning_rate = learning_rate 
		self.no_of_itr = no_of_itr 

	def fit(self, hdl, totl): 

		self.m, self.n = hdl.shape	  
		self.w = np.zeros((self.n, 1)) 
		self.b = 0
		self.hdl = hdl
		self.totl = totl 

		for i in range(self.no_of_itr): 
			self.update_weigths() 

	def update_weigths(self): 
		Y_prediction = self.predict(self.hdl) 

		dw = -(self.hdl.T).dot(self.totl - Y_prediction)/self.m 

		db = -np.sum(self.totl - Y_prediction)/self.m 

		self.w = self.w - self.learning_rate * dw 
		self.b = self.b - self.learning_rate * db 

	def predict(self, hdl): 
		return hdl.dot(self.w) + self.b 

	def print_weights(self): 
		print('Weights for the respective features are :') 
		print(self.w) 
		print() 

		print('Bias value for the regression is ', self.b) 

model = Linear_Regression(learning_rate=0.03, 
                          no_of_itr=2000) 
model.fit(hdl, totl) 

plt.scatter(df['hdlngth'], df['totlngth']) 
plt.xlabel('Possum Head Length') 
plt.ylabel('Possum Total Length') 
plt.title('Head Length v/s Total Length') 
  
X = df['hdlngth'].values 
plt.plot(X, 1 * X + 2) 
plt.show() 