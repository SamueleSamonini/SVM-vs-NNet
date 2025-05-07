from common.load_data import load_fashion_mnist

x_train, y_train, x_test, y_test = load_fashion_mnist()

print("Train set:", x_train.shape, y_train.shape)
print("Test set:", x_test.shape, y_test.shape)
print("Pixel value range:", x_train.min(), "-", x_train.max())

from svm.kernel_functions import linear_kernel, rbf_kernel, polynomial_kernel
import numpy as np

X = np.array([[1, 2], [3, 4]])
Y = np.array([[5, 6]])

print("Linear:\n", linear_kernel(X, Y))
print("RBF:\n", rbf_kernel(X, Y, gamma=0.1))
print("Poly:\n", polynomial_kernel(X, Y, degree=2))
