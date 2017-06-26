import matplotlib.pyplot as plt
import numpy as np

xRaw = [1, 5, 10, -1]
yRaw = [2, 6, 4, 1]

def f(x, X):
	#Function to graph that fits the data
	return X[0,0] + (x * X[1,0]) + (X[2,0] * x ** 2) + (X[3,0] * x ** 3)

#System equations
#y(n) = a + b * x(n) + c * x(n)^2 + d * x(n)^3
#y(n) = a + b * x(n) + c * x(n)^2 + d * x(n)^3
#y(n) = a + b * x(n) + c * x(n)^2 + d * x(n)^3
#y(n) = a + b * x(n) + c * x(n)^2 + d * x(n)^3

# XA = B
A = np.matrix([[1, xRaw[0], xRaw[0] ** 2, xRaw[0] ** 3],
			  [1, xRaw[1], xRaw[1] ** 2, xRaw[1] ** 3],
			  [1, xRaw[2], xRaw[2] ** 2, xRaw[2] ** 3],
			  [1, xRaw[3], xRaw[3] ** 2, xRaw[3] ** 3]])

B = np.matrix(yRaw)

# X = B * A^-1
X = np.linalg.inv(A)
# print X, "\n\n"
# print B, "\n\n"
X = np.dot(X, np.transpose(B))

print X, "\n\n"

plt.figure(1)
plt.scatter(xRaw, yRaw)
plt.ylabel('Data')

x = np.arange(-2.0, 12.0, 0.1)

plt.plot(x, f(x, X))

plt.show()