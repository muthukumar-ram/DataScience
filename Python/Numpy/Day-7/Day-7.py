# -*- coding: utf-8 -*-
"""


@author: muthukumar_ram

Day-7
"""
import numpy as np

# File Input and Output with Arrays
#np.save and np.load
arr1 = np.arange(1,10).reshape(3,3)
np.savetxt('nparray.txt',arr1,delimiter='|')
arr2 = np.loadtxt('nparray.txt',delimiter='|')
print(arr2)

#save
np.save('nparray',arr1) # it will add .npy extension
arr3 = np.load('nparray.npy')
print(arr3)
np.savez('nparray.npz',a=arr1,b=arr2) # it will create dict like array
archive = np.load('nparray.npz')
print(archive['b'])
"""
print(archive['b'])
[[ 1.  2.  3.]
 [ 4.  5.  6.]
 [ 7.  8.  9.]]
"""
# similar to np.savez()
# np.savez_compressed() https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez_compressed.html
"""
Linear Algebra
We have seen few linear algebra methods and attributes like inverse,dot 
Now we are going to try least squre calculation for linear regression problem
"""
#lstsq

# line y = mx+b    
X = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])

from matplotlib import pyplot as plt
plt.plot(X,y)
plt.show()
A = np.vstack([X, np.ones(len(X))]).T # setting y^ to 1for examining slopes
m,c = np.linalg.lstsq(A, y)[0]
print(m, c)

plt.plot(X, y, 'o', label='Original data', markersize=10)
plt.plot(X, m*X + c, 'r', label='Predicted line')
plt.legend()
plt.show()

"""
Solving Squared Matrix equation by using solve method
3 * x0 + x1 = 9 and x0 + 2 * x1 = 8
"""
#coefficient of first eq is (3,1) and 1,2 is second 1

a = np.array([[3,1],[1,2]])
b = np.array([9,8])
x = np.linalg.solve(a,b)
print(x)
# [ 2.  3.]
# validated manually 3*2+3 = 9 and 2 + 2*3 = 8
# validate using allclose
np.allclose(np.dot(a,x),b)
# Out[30]: True

# random.seed()

np.random.seed(0) # makes the random numbers predictable
np.random.rand(4)
np.random.rand(4)
"""
np.random.rand(4)
Out[31]: array([ 0.2271475 ,  0.56174516,  0.09327733,  0.54818089])


np.random.rand(4)
Out[32]: array([ 0.54314201,  0.31566946,  0.04276932,  0.84404385])
we got diffent op

"""
np.random.seed(0)
np.random.rand(4)
np.random.seed(0)
np.random.rand(4)
"""
np.random.seed(0)

np.random.rand(4)
Out[36]: array([ 0.5488135 ,  0.71518937,  0.60276338,  0.54488318])

np.random.seed(0)

np.random.rand(4)
Out[38]: array([ 0.5488135 ,  0.71518937,  0.60276338,  0.54488318])

We got the same array
(pseudo-)random numbers work by starting with a number (the seed),
multiplying it by a large number, then taking modulo of that product.
The resulting number is then used as the seed to generate the next "random" number. 
When you set the seed (every time), it does the same thing every time, giving you the same numbers.
"""



