import matplotlib.pyplot as plt
import numpy as np

xRaw = [1, 5, 10, -1, 7, 8]
yRaw = [2, 6, 4, 1, 3, 9]

POLYNOMIAL = 5

def f(x, X):
	#Function to graph that fits the data
	sum = 0;
	for n in range(0, POLYNOMIAL + 1):
		sum += X[n, 0] * (x ** n)
	return sum

#System equations
#y(n) = a + b * x(n) + c * x(n)^2 + d * x(n)^3
#y(n) = a + b * x(n) + c * x(n)^2 + d * x(n)^3
#y(n) = a + b * x(n) + c * x(n)^2 + d * x(n)^3
#y(n) = a + b * x(n) + c * x(n)^2 + d * x(n)^3

# XA = B
A = np.zeros((POLYNOMIAL + 1, POLYNOMIAL + 1))

for row in range(0, POLYNOMIAL + 1):
	for n in range(0, POLYNOMIAL + 1):
		A[row, n] = (xRaw[row] ** n)

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