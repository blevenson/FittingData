import matplotlib.pyplot as plt
import numpy as np
import math

RESOLUTION = 0.01
ROBOT_WIDTH = 1.0;
INNER_SPACING = ROBOT_WIDTH / 2

class Waypoint():

	def __init__(self, x_, y_, heading_):
		self.x = x_
		self.y = y_
		#Convert heading from degrees to slope
		self.slope = math.tan(math.radians(heading_))
		self.heading = heading_

#Class for each spline section
class Spline():
	def __init__(self, valueMatrix_, startPoint_, endPoint_):
		self.valueMatrix = valueMatrix_
		self.startPoint = startPoint_
		self.endPoint = endPoint_

waypoints = []
# waypoints.append(Waypoint(0, 0, 45))
# waypoints.append(Waypoint(1, 5, 30.0))
# waypoints.append(Waypoint(2, 2, 0.0))
# waypoints.append(Waypoint(8, 8, -45))

waypoints.append(Waypoint(0, 0, 0))
waypoints.append(Waypoint(5, 0, 0))
waypoints.append(Waypoint(10, 0, 0))
waypoints.append(Waypoint(15, 0, 45))
waypoints.append(Waypoint(17, 2, 45))

#Length is 1 less than amount of points
#Stores an array of the constants for each spline
splines = []
leftSplines = []
rightSplines = []

leftPoints = []
rightPoints = []

def f(x, X):
	#Function to graph that fits the data
	return X[0,0] + (x * X[1,0]) + (X[2,0] * x ** 2) + (X[3,0] * x ** 3)

def df(x, X):
	#Function to graph the heading
	return X[1,0] + 2*(X[2,0] * x) + 3*(X[3,0] * x ** 2)

def parallelCurveY(x1, X, d):
	return f(x1, X) - (d / ((1 + df(x1, X) ** 2) ** (0.5)))

def parallelCurveX(x, d, X):
	return x + (d * df(x, X)) / ((1 + df(x, X) ** 2) ** 0.5)

#Plots the spline
def plotSpline(spline):
	#plt.plot(x, f(x, X))

	x = np.arange(spline.startPoint.x, spline.endPoint.x + RESOLUTION, RESOLUTION)

	plt.plot(x, f(x, spline.valueMatrix))

#Plots the spline's heading
def plotHeading(spline):
	x = np.arange(spline.startPoint.x, spline.endPoint.x + RESOLUTION, RESOLUTION)

	plt.plot(x, df(x, spline.valueMatrix))

#Plots parallel curve - Deprecated
def plotParallelCurve(section, startingPoint, endingPoint, d):
	x1 = np.arange(startingPoint.x, endingPoint.x + RESOLUTION, RESOLUTION)
	x2 = parallelCurveX(x1, d, section)

	plt.plot(x2, parallelCurveY(x1, section, d))

def calcParallelCurvePoint(x, valueMatrix, d):
	x1 = parallelCurveX(x, d, valueMatrix)

	return Waypoint(x1, parallelCurveY(x, valueMatrix, d), 0)

def calcLengthCurve(spline):
	length = 0

	xValues = np.arange(spline.startPoint.x, spline.endPoint.x + RESOLUTION, RESOLUTION)
	yValues = f(xValues, spline.valueMatrix)
	for i in range(len(xValues) - 1):
		length += ((xValues[i + 1] - xValues[i]) ** 2 + (yValues[i + 1] - yValues[i]) ** 2) ** 0.5

	return length

# Returns the length of a curve given an array of Waypoints on the curve
def calcDistOfPoints(points):
	length = 0

	previousPoint = points[0]
	for point in points[1:]:

		# Calc length between points using everyone's favorite distance formula
		length += (((point.x - previousPoint.x) ** 2) + ((point.y - previousPoint.y) ** 2)) ** 0.5

		# Update the previous point to be the current point
		previousPoint = point

	return length


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

	B = np.matrix([previousPoint.y, currentPoint.y, previousPoint.slope, currentPoint.slope])

	# X = B * A^-1
	X = np.linalg.inv(A)
	# print X, "\n\n"
	# print B, "\n\n"
	X = np.dot(X, np.transpose(B))
	return Spline(X, previousPoint, currentPoint)

xRaw = [point.x for point in waypoints]
yRaw = [point.y for point in waypoints]


plt.figure(1)
plt.scatter(xRaw, yRaw)
plt.ylabel('Data')

#Calculate the splines between waypoints and add to splines array
for i in range(1, len(waypoints)):
	splines.append(calculateSpline(waypoints[i - 1], waypoints[i]))

	'''
	DO NOT USE.  PATHS ARE NOT GUARENTEED TO BE PARALLEL TO DESIRED PATH
		- RESULTS IN LEFT AND RIGHT WHEEL ACTING INDEPENDENTLY

	#Calculate the left drive path
	leftSplines.append(calculateSpline(Waypoint((waypoints[i - 1].x - INNER_SPACING*math.sin(math.radians(waypoints[i - 1].heading))), (waypoints[i - 1].y + INNER_SPACING*math.cos(math.radians(waypoints[i - 1].heading))), waypoints[i - 1].heading),
		Waypoint((waypoints[i].x - INNER_SPACING*math.sin(math.radians(waypoints[i].heading))), (waypoints[i].y + INNER_SPACING*math.cos(math.radians(waypoints[i].heading))), waypoints[i].heading)
		))

	#Calculate the right drive path
	rightSplines.append(calculateSpline(Waypoint((waypoints[i - 1].x + INNER_SPACING*math.sin(math.radians(waypoints[i - 1].heading))), (waypoints[i - 1].y - INNER_SPACING*math.cos(math.radians(waypoints[i - 1].heading))), waypoints[i - 1].heading),
		Waypoint((waypoints[i].x + INNER_SPACING*math.sin(math.radians(waypoints[i].heading))), (waypoints[i].y - INNER_SPACING*math.cos(math.radians(waypoints[i].heading))), waypoints[i].heading)
		))
	'''

totalSplineLength = 0
leftSplineLength = 0
rightSplineLength = 0

for spline in splines:
	for x in np.arange(spline.startPoint.x, spline.endPoint.x + RESOLUTION, RESOLUTION):
		leftPoints.append(calcParallelCurvePoint(x, spline.valueMatrix, INNER_SPACING))
		# print(calcParallelCurvePoint(np.arange(spline.startPoint.x, spline.endPoint.x + RESOLUTION, RESOLUTION), spline.valueMatrix, INNER_SPACING))
		rightPoints.append(calcParallelCurvePoint(x, spline.valueMatrix, -INNER_SPACING))

#plot splines
for i in range(0, len(waypoints) - 1):
	plotSpline(splines[i])
	# plotSpline(leftSplines[i])
	# plotSpline(rightSplines[i])
	# plotParallelCurve(splines[i].valueMatrix, waypoints[i], waypoints[i + 1], INNER_SPACING)
	# plotParallelCurve(splines[i].valueMatrix, waypoints[i], waypoints[i + 1], -INNER_SPACING)
	# plotHeading(splines[i])

	#Calculate length of splines
	totalSplineLength += calcLengthCurve(splines[i])

#Plot parallel curves
leftXRaw = [point.x for point in leftPoints]
leftYRaw = [point.y for point in leftPoints]
rightXRaw = [point.x for point in rightPoints]
rightYRaw = [point.y for point in rightPoints]

plt.plot(leftXRaw, leftYRaw)
plt.plot(rightXRaw, rightYRaw)

leftSplineLength = calcDistOfPoints(leftPoints)
rightSplineLength = calcDistOfPoints(rightPoints)

print "Length: ", totalSplineLength
print "LeftLength: ", leftSplineLength
print "RightLength: ", rightSplineLength

plt.show()
