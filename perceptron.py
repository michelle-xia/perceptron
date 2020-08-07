import numpy as np

# Initialize NAND Gate
# x0 is dummy variable for bias term
#     x0  x1  x2
x = [[1., 0., 0.],
     [1., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.]]

y =[1.,
    1.,
    1.,
    0.]

# Initialize weights
w = np.zeros(len(x[0]))

# Calculate Dot Product
f = np.dot(w, x[0])

# Activation Function
z = 0.0
if f > z:
    yhat = 1.
else:
    yhat = 0.

# Update weights
eta = 0.1
w[0] = w[0] + eta * (y[0] - yhat) * x[0][0]
w[1] = w[1] + eta * (y[0] - yhat) * x[0][1]
w[2] = w[2] + eta * (y[0] - yhat) * x[0][2]