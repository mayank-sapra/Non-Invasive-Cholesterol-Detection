import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay

# Load your dataset (ensure the correct path is used)
dataset = pd.read_csv('Sheet1-Table 1.csv')  # replace with your dataset path
X = dataset.iloc[:, :-1].values  # Features (assuming the last column is the label)
y = dataset.iloc[:, -1].values   # Label

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)

# SVM (Support Vector Machine)
svm = SVC(kernel='rbf', probability=True, random_state=0)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# k-NN (k-Nearest Neighbors)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

# Display accuracies
print(f"Logistic Regression Accuracy: {log_reg_accuracy:.2f}")
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"k-NN Accuracy: {knn_accuracy:.2f}")

# Cross-validation for Logistic Regression
log_reg_cv_scores = cross_val_score(log_reg, X, y, cv=5)
print(f"Logistic Regression CV Accuracy: {log_reg_cv_scores.mean():.2f}")

# Cross-validation for SVM
svm_cv_scores = cross_val_score(svm, X, y, cv=5)
print(f"SVM CV Accuracy: {svm_cv_scores.mean():.2f}")

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"Random Forest CV Accuracy: {rf_cv_scores.mean():.2f}")

# Cross-validation for k-NN
knn_cv_scores = cross_val_score(knn, X, y, cv=5)
print(f"k-NN CV Accuracy: {knn_cv_scores.mean():.2f}")

# Plot confusion matrix
def plot_conf_matrix(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(title)
    plt.show()

# Confusion Matrices
plot_conf_matrix(log_reg, X_test, y_test, 'Logistic Regression Confusion Matrix')
plot_conf_matrix(svm, X_test, y_test, 'SVM Confusion Matrix')
plot_conf_matrix(rf, X_test, y_test, 'Random Forest Confusion Matrix')
plot_conf_matrix(knn, X_test, y_test, 'k-NN Confusion Matrix')

# ROC Curves (for Logistic Regression, SVM, Random Forest)
def plot_roc_curve(models, model_names, X_test, y_test):
    plt.figure(figsize=(10, 6))
    for model, name in zip(models, model_names):
        if hasattr(model, "predict_proba"):
            y_pred_prob = model.predict_proba(X_test)[:, 1]
        else:  # for SVC without predict_proba (use decision function instead)
            y_pred_prob = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.show()

# Models to include in ROC Curve plotting
models = [log_reg, svm, rf]
model_names = ['Logistic Regression', 'SVM', 'Random Forest']
plot_roc_curve(models, model_names, X_test, y_test)

# Checking for overfitting by comparing training and test accuracies
train_accuracies = [accuracy_score(y_train, log_reg.predict(X_train)),
                    accuracy_score(y_train, svm.predict(X_train)),
                    accuracy_score(y_train, rf.predict(X_train)),
                    accuracy_score(y_train, knn.predict(X_train))]

test_accuracies = [log_reg_accuracy, svm_accuracy, rf_accuracy, knn_accuracy]

# Plotting training vs. testing accuracies
labels = ['Logistic Regression', 'SVM', 'Random Forest', 'k-NN']
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, train_accuracies, width, label='Training Accuracy')
plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Training vs Test Accuracy')
plt.xticks(x, labels)
plt.legend()
plt.show()
