import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data cleaning
df = pd.read_csv("data/breast-cancer-dataset.csv", names=[
"id number",
"Clump Thickness",
"Uniformity of Cell Size",
"Uniformity of Cell Shape",
"Marginal Adhesion",
"Single Epithelial Cell Size",
"Bare Nuclei",
"Bland Chromatin",
"Normal Nucleoli",
"Mitoses",
"Class"
])

print(df.head(5))
df = df.replace('?', np.NaN)
# print(df.isna().sum())

X = df[["Clump Thickness",
"Uniformity of Cell Size",
"Uniformity of Cell Shape",
"Marginal Adhesion",
"Single Epithelial Cell Size",
"Bare Nuclei",
"Bland Chromatin",
"Normal Nucleoli",
"Mitoses"
]].values.astype(np.float32)

idx = np.where(np.isnan(X))
X[idx] = np.take(np.nanmedian(X, axis = 0), idx[1])

y = df["Class"].values
y = np.array(y==4, dtype=np.float32)

# Adding bias factor
X = np.hstack((np.ones((len(X), 1)), X))

m, n = X.shape
K = 2

W = np.zeros(n)


def g(z):    # sigmoid activation function
	return 1/(1+np.exp(-z))

def h(X, W):     
	return g(X @ W)

def J(predictions, y):     # loss function
	return (1/m) * (-y @ np.log(predictions) -(1-y) @ np.log(1-predictions))

def get_gradient(W, X, y):
	predictions = h(X, W)
	gradient = (1/m) * X.T @ (predictions - y)
	return gradient


hist = {'loss': [], 'acc': []}
alpha = 0.1

for i in range(100):
	gradient = get_gradient(W, X, y)
	W -= alpha * gradient

	# loss
	predictions = h(X, W)
	loss = J(predictions, y)
	hist['loss'].append(loss)

	# acc
	c = 0
	for j in range(len(y)):
		if (h(X[j], W) > .5) == y[j]:
			c += 1

	acc = c / len(y)
	hist['acc'].append(acc)

	# print stats
	if i % 10 == 0: print(loss, acc)


f = plt.figure(figsize=(8, 3))
plt.suptitle("Performance of Classifier Model", y=1)
ax = f.add_subplot(1, 2, 1)
ax.plot(hist['loss'])
ax.set_xlabel('loss')
ax2 = f.add_subplot(1, 2, 2)
ax2.plot(hist['acc'])
ax2.set_xlabel('accuracy')
plt.tight_layout()
plt.show()