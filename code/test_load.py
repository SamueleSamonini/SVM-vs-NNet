from common.load_data import load_fashion_mnist

x_train, y_train, x_test, y_test = load_fashion_mnist()

print("Train set:", x_train.shape, y_train.shape)
print("Test set:", x_test.shape, y_test.shape)
print("Pixel value range:", x_train.min(), "-", x_train.max())
