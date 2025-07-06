# Import necessary libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split from sklearn.linear_model import LogisticRegression from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the digits dataset
digits = load_digits()

# Features (X) and Target (y)
X = digits.data
y = digits.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
logistic_model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto') logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# SVM Model (with RBF kernel)
svm_model = SVC(kernel='rbf', gamma='scale', C=1.0) svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Calculate accuracies
logistic_accuracy = accuracy_score(y_test, y_pred_logistic) svm_accuracy = accuracy_score(y_test, y_pred_svm)

# Print results
print(f"Logistic Regression Accuracy: {logistic_accuracy * 100:.2f}%") print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")

# Confusion Matrix and Classification Report for Logistic Regression
print("\nConfusion Matrix (Logistic Regression):") print(confusion_matrix(y_test, y_pred_logistic))
print("\nClassification Report (Logistic Regression):") print(classification_report(y_test, y_pred_logistic))

# Confusion Matrix and Classification Report for SVM
print("\nConfusion Matrix (SVM):")
print(confusion_matrix(y_test, y_pred_svm)) print("\nClassification Report (SVM):")
print(classification_report(y_test, y_pred_svm))
