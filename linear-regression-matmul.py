import numpy as np
import matplotlib.pyplot as plt
import random


def gen_data():

	np.random.seed(20)  # making values reproducible
	random.seed(10)
	f = 1
	num_points = 20
	x = np.arange(0, 1, 0.01)
	y = np.sin(2*np.pi*f*x)

	x_noisy = random.sample(list(x), num_points)
	x_noisy.sort()

	y_noisy = []
	for i in x_noisy:
		y_noisy.append(np.sin(2*np.pi*f*i) + np.random.normal(0, 0.1))
	return x_noisy, y_noisy


def plot_data():

	# plt.plot(x,y, label="sin wave")
	plt.plot(x, y, "*r", label="sin wave with noise")
	# plt.plot(x, Y_predicted(x), label="estimated function")
	plt.title("Point Generation Visualisation")
	plt.legend()
	plt.show()





x, y = gen_data()
dim = len(x)
x = np.array(x)
X = np.stack((np.ones((dim)), x, x**2, x**3, x**4))
X = X.T

def h(X, theta):     # prediction for each point
	return X.dot(theta)

theta = np.random.rand(5)
predictions = h(X, theta)

def J(theta, X, y):
	return np.mean(np.square(h(X, theta) - y))

losses = []
alpha = 1.4
for _ in range(50000):
	theta = theta - alpha*(1/dim) * (X.T @ ((X @ theta) - y))
	losses.append(J(theta, X, y))

predictions = h(X, theta)
print(theta)
print(losses[-1])
print(predictions)

print(X.shape)
print(len((X @ theta)-y))


plt.plot(X[:,1], predictions, label="predictions")
plt.plot(X[:,1], y, "rx", label = "labels")
# plt.plot(X[:,1], predictions, label="predictions")
plt.legend()
plt.show()