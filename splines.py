import numpy as np
import matplotlib.pyplot as plt
from numpy import double

#   mathematical spline definition
def spline(Z, H, X, C, D, i, x):
    return (Z[i] / (6*H[i])) * ((X[i+1] - x)**3) + (Z[i+1] / (6*H[i])) * ((x - X[i])**3) + C[i]*(x-X[i]) + D[i]*(X[i+1]-x)


#   print spline
def printSpline(i):
    print('spline', i, ': ', Z[i] / (6*H[i]), '(', X[i + 1], '-x)^3+', Z[i+1] / (6*H[i]), '(x-', X[i], ')^3+', C[i],
          '(x-', X[i], ')+', D[i], '(', X[i + 1], '-x)\n')


#   sort x,y arrays by x values
def sortPoints(x, y):
    for i in range(0, len(x)-1):
        for j in range(0, len(x)-1):
            if x[j]>x[j+1]:
                x[j+1] += x[j]
                x[j] = x[j+1]-x[j]
                x[j+1] -= x[j]
                y[j+1] += y[j]
                y[j] = y[j+1] - y[j]
                y[j+1] -= y[j]


#   ask for user input
n = int(input("Enter number of points : ")) -1
X = np.zeros(n+1)
Y = np.zeros(n+1)

for i in range(0, n+1):
    st = "please give me point #%d\nx = " % (i+1)
    X[i] = double(input(st))
    Y[i] = double(input("y = "))

sortPoints(X, Y)


#   initialize our variables
H = np.zeros(n+1)
Z = np.zeros(n+1)
C = np.zeros(n+1)
D = np.zeros(n+1)

#   calculate H[i]
for i in range(0, n):
    H[i] = X[i + 1] - X[i]

#   initialize our matrix, and result vector
M = np.zeros((n-1, n-1))
r = np.zeros((n-1, 1))

#   fill up our matrix
for i in range(0, len(M)):
    temp = len(M[i])
    for j in range(0, temp):
        if i == j:
            M[i][j] = 2*(H[i]+H[i+1])
        if i == j-1:
            M[i][j] = H[j]
        if i == j+1:
            M[i][j] = H[i]

#   fill up our result vector
for i in range (n-1):
    r[i][0] = (6/H[i+1])*(Y[i+2]-Y[i+1])-(6/H[i])*(Y[i+1]-Y[i])

#   calculate Z
Z_sub = np.linalg.solve(M, r)

#   fill Z in original array
for i in range (n-1):
    Z[i + 1] = Z_sub[i][0]

#   calculate C[i] and D[i]
for i in range(0, n):
    D[i] = Y[i] / H[i] - Z[i] * H[i] / 6
    C[i] = Y[i + 1] / H[i] - Z[i + 1] * H[i] / 6


#   initialize splines matrix
x_splines = [[0 for x in range(10)] for y in range(n)]
y_splines = [[0 for x in range(10)] for y in range(n)]

#   choose 10 points in every spline
for i in range(0, n):
    x_splines[i] = np.linspace(X[i], X[i+1], 10)
for i in range(0, n):
    y_splines[i] = np.zeros(10)

#   calculate Y[i] for every spline
for i in range(0, n):
    for j in range(0, 10):
        y_splines[i][j] = spline(Z, H, X, C, D, i, x_splines[i][j])

#   show points of splines on graph
for i in range(0, n):
    plt.scatter(x_splines[i], y_splines[i], color='black')

#   show original input points
plt.scatter(X, Y, color='red')

#   show the graph
for i in range(0, n):
    plt.plot(x_splines[i], y_splines[i], color='green')

#   find y's range
y_min = Y[0]
y_max = Y[0]
for i in range (0, n+1):
    if Y[i]>y_max:
        y_max = Y[i]
    if Y[i]<y_min:
        y_min = Y[i]

#   print splines
for i in range(0, n):
    printSpline(i)

#   define limits, labels, and show graph
plt.ylim(y_min-(y_max-y_min)*0.2, y_max+(y_max-y_min)*0.2)
plt.xlim(X[0]-(X[n]-X[0])*0.2, X[n]+(X[n]-X[0])*0.2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic Spline')
plt.show()
