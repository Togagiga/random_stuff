import numpy as np
import matplotlib.pyplot as plt
import random


def gen_data(num_points):

	np.random.seed(20)  # making values reproducible
	random.seed(10)
	f = 1
	# num_points = 30
	x = np.arange(0, 1, 0.01)
	y = np.sin(2*np.pi*f*x)

	x_noisy = random.sample(list(x), num_points)
	x_noisy.sort()

	y_noisy = []
	for i in x_noisy:
		y_noisy.append(np.sin(2*np.pi*f*i) + np.random.normal(0, 0.1))
	return x, y,x_noisy, y_noisy


def plot_data():

	plt.plot(x,y, label="sin wave")
	plt.plot(x_noisy, y_noisy, "*r", label="noisy samples")
	plt.plot(x, Y_predicted(x, W), label="estimated function")
	plt.title("Point Generation Visualisation")
	plt.legend()
	plt.show()


def init_weights(order):       # order of the polynomial model
	
	W = np.random.rand(order+1)
	return W


def Y_predicted(x, W):

	y_pred = 0
	for term in range(len(W)):
		y_pred += W[term]*np.power(x,term)
	return y_pred


def MSE(W, x, y):
	
	J = 0
	for i in range(len(x)):
		J += (Y_predicted(x[i], W)-y[i])**2
	J = J/len(x)
	return J


def GD_step(W, x, y, alpha):

	grad = np.zeros(len(W))
	for i in range(len(x)):
		lst = np.zeros(len(W))
		for j in range(len(W)):
			lst[j] = np.power(x[i], j)

		grad += (Y_predicted(x[i], W) - y[i])*lst

	W = W - (alpha/len(x))*grad
	return W


### SETTINGS ###
alpha = 1.4
num_points = 20
order_poly = 4
epochs = 5000
################

'''
good start values: 1.4, 20, 4, 10000

NOTE: if order_ploy > num_points we get overfitting as less independent data than unknowns (always need more training samples than features)
'''

x, y, x_noisy, y_noisy = gen_data(num_points)
W = init_weights(order_poly)

for step in range(epochs):
	J = MSE(W, x_noisy, y_noisy)
	W = GD_step(W, x_noisy, y_noisy, alpha)

print(f"Loss: {J}")
print(f"Weights: {W}")
plot_data()