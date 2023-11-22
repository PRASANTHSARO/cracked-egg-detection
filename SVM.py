import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import os
import cv2 

def load_and_preprocess_custom_dataset(data_dir, image_size=(8, 8)):
    data = []
    labels = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            try:
                label_int = int(label)
            except ValueError:
                print(f"Skipping invalid label: {label}")
                continue
            for image_file in os.listdir(label_path):   
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)
                data.append(image.flatten())
                labels.append(label_int)
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def image_classification_svm(X_train, X_test, y_train, y_test, kernel_type='rbf'):
    # Create the SVM classifier with specified kernel type
    if kernel_type == 'linear':
        clf = svm.SVC(kernel='linear', probability=True)
    elif kernel_type == 'poly':
        clf = svm.SVC(kernel='poly', degree=3, probability=True)  # You can adjust the degree
    elif kernel_type == 'sigmoid':
        clf = svm.SVC(kernel='sigmoid', probability=True)
    else:
        clf = svm.SVC(kernel='rbf', probability=True)  # Default to radial basis function (RBF) kernel

    # Fit the model
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

    # ROC Curve and AUC
    probas_ = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {kernel_type} Kernel')
    plt.legend(loc="lower right")
    plt.show()

    # Plot the "loss curve" (decision function values)
    decision_values_train = clf.decision_function(X_train)
    plt.plot(decision_values_train, label=f'Decision Function Values (Training) - {kernel_type} Kernel')
    plt.xlabel('Sample Index')
    plt.ylabel('Decision Function Value')
    plt.legend()
    plt.title(f'Training Loss Curve - {kernel_type} Kernel')
    plt.show()

    # Print the kernel type and accuracy
    print(f"Kernel: {kernel_type}")
    print("Accuracy:", accuracy)

    if kernel_type == 'linear':
        # For linear kernel, we can also extract feature coefficients
        feature_importance = clf.coef_.flatten()
        print("Feature Importance (Coefficients for a Linear Kernel):")
        print(feature_importance)

    # Generate the classification report
    report = classification_report(y_test, y_pred)
    print(report)

    return fpr, tpr, roc_auc, decision_values_train, confusion_mat, accuracy

if __name__ == "__main__":
    data_dir = "/kaggle/input/broken-eggs/train"
    X_train, X_test, y_train, y_test = load_and_preprocess_custom_dataset(data_dir, image_size=(8, 8))

    # Test with different SVM kernels
    kernel_types = ['linear', 'poly', 'sigmoid', 'rbf']
    plt.figure(figsize=(15, 12))

    for kernel_type in kernel_types:
        fpr, tpr, roc_auc, decision_values_train, confusion_mat, accuracy = image_classification_svm(
            X_train, X_test, y_train, y_test, kernel_type)

        # Display and store performance metrics for each kernel
        print(f"\nPerformance Metrics for {kernel_type} Kernel:")
        print("Confusion matrix:")
        print(confusion_mat)
        print(f"Accuracy: {accuracy}")
        print(f"AUC: {roc_auc}")

    plt.show()
