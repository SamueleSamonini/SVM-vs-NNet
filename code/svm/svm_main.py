import numpy as np
from common.load_data import load_fashion_mnist
from svm.svm_dual_solver import KernelSVM
from svm.kernel_functions import linear_kernel, rbf_kernel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
print("Loading data...")
x_train, y_train, x_test, y_test = load_fashion_mnist()

# For simplicity: binary classification (e.g., class 0 vs class 1)
class_1, class_2 = 0, 1
train_filter = np.where((y_train == class_1) | (y_train == class_2))
test_filter = np.where((y_test == class_1) | (y_test == class_2))

x_train_bin, y_train_bin = x_train[train_filter], y_train[train_filter]
x_test_bin, y_test_bin = x_test[test_filter], y_test[test_filter]

# Set labels to -1 and +1
y_train_bin = np.where(y_train_bin == class_1, -1, 1)
y_test_bin = np.where(y_test_bin == class_1, -1, 1)

# Define kernel and SVM
print("Training SVM with RBF kernel...")
model = KernelSVM(kernel=rbf_kernel, C=1.0, gamma=0.05)
model.fit(x_train_bin, y_train_bin)

# Predict
print("Predicting...")
y_pred = model.predict(x_test_bin)

# Evaluation
acc = accuracy_score(y_test_bin, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_bin, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test_bin, y_pred))
