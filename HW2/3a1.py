import scipy.io as scio
import scipy.optimize.minimize as minimize
import numpy as np
data_file = './hw2files/occam1.mat'

data = scio.loadmat(data_file)

print()
x = data['x']
y = data['y']
y = np.array(y)
print(y)

def loss(theta, alpha):
    return 0




# def kernel(x):
    # return x + x**1

# print(x, y)