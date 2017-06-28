import matplotlib.pyplot as plt
import numpy as np
import math

RESOLUTION = 0.01

class Waypoint():

	def __init__(self, x_, y_, heading_):
		self.x = x_
		self.y = y_
		#Convert heading from degrees to slope
		self.heading = math.tan(math.radians(heading_))

waypoints = []
# waypoints.append(Waypoint(0, 0, 45))
# waypoints.append(Waypoint(1, 5, 30.0))
# waypoints.append(Waypoint(2, 2, 0.0))
# waypoints.append(Waypoint(8, 8, -45))

waypoints.append(Waypoint(0, 0, 0))
waypoints.append(Waypoint(5, 0, 0))
waypoints.append(Waypoint(10, 0, 0))
# waypoints.append(Waypoint(15, 0, 0))
waypoints.append(Waypoint(17, 8, 0))

#Length is 1 less than amount of points
#Stores an array of the constants for each spline
splines = []

def f(x, X):
	#Function to graph that fits the data
	return X[0,0] + (x * X[1,0]) + (X[2,0] * x ** 2) + (X[3,0] * x ** 3)

def df(x, X):
	#Function to graph the heading
	return X[1,0] + 2*(X[2,0] * x) + 3*(X[3,0] * x ** 2)

#Plots the spline
def plotSpline(section, startingPoint, endingPoint):
	#plt.plot(x, f(x, X))

	x = np.arange(startingPoint.x, endingPoint.x + RESOLUTION, RESOLUTION)

	plt.plot(x, f(x, section))

#Plots the spline's heading
def plotHeading(section, startingPoint, endingPoint):
	x = np.arange(startingPoint.x, endingPoint.x + RESOLUTION, RESOLUTION)

	plt.plot(x, df(x, section))

#Points are waypoints
def calculateSpline(previousPoint, currentPoint):
	#System equations:
	# y(start) = a + b * x(start) + c * x(start)^2 + d * x(start)^3
	# y(end) = a + b * x(end) + c * x(end)^2 + d * x(end)^3
	# heading(start) -> slope = b + ((2*c) * x(start)) + ((3d) * x(start)^2)
	# heading(end) -> slope = b + ((2*c) * x(end)) + ((3d) * x(end)^2)

	A = np.matrix([[1, previousPoint.x, previousPoint.x ** 2, previousPoint.x ** 3],
				  [1, currentPoint.x, currentPoint.x ** 2, currentPoint.x ** 3],
				  [0, 1, previousPoint.x * 2, 3 * previousPoint.x ** 2],
				  [0, 1, currentPoint.x * 2, 3 * currentPoint.x ** 2]])

	B = np.matrix([previousPoint.y, currentPoint.y, previousPoint.heading, currentPoint.heading])

	# X = B * A^-1
	X = np.linalg.inv(A)
	# print X, "\n\n"
	# print B, "\n\n"
	X = np.dot(X, np.transpose(B))
	return X

xRaw = [point.x for point in waypoints]
yRaw = [point.y for point in waypoints]


plt.figure(1)
plt.scatter(xRaw, yRaw)
plt.ylabel('Data')

#Calculate the splines between waypoints and add to splines array
for i in range(1, len(waypoints)):
	splines.append(calculateSpline(waypoints[i - 1], waypoints[i]))

#plot splines
for i in range(0, len(waypoints) - 1):
	plotSpline(splines[i], waypoints[i], waypoints[i + 1])
	# plotHeading(splines[i], waypoints[i], waypoints[i + 1])


plt.show()
