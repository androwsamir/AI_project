# Model Evaluating
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def model_evaluation(y_pred_log, y_pred_svm, y_pred_decision, y_test):
    # Making the Confusion Matrix for Logistic Regression
    cm_log = confusion_matrix(y_test, y_pred_log)
    
    # Making the Confusion Matrix for SVM
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    
    # Making the Confusion Matrix for DecisionTree
    cm_decision = confusion_matrix(y_test, y_pred_decision)
    
    #Model Accuracy : how often is the classifier correct?
    #Model Accuracy for Logistic Regression
    logistic_regression_accuracy = metrics.accuracy_score(y_test, y_pred_log)
    
    #Model Accuracy for SVM
    svm_accuracy = metrics.accuracy_score(y_test, y_pred_svm)
    
    #Model Accuracy for DecisionTree
    DecisionTree_accuracy = metrics.accuracy_score(y_test, y_pred_decision)
    
    return logistic_regression_accuracy, svm_accuracy, DecisionTree_accuracy, cm_log, cm_svm, cm_decision