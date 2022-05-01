# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import Preprocessing
import Classification
import Model_Evaluation

def main():    
    x_train , x_test , y_train , y_test = Preprocessing.preprocessing()
    y_pred_log, y_pred_svm, y_pred_decision, classifier_log, classifier_svm, classifier_decision = Classification.classification(x_train , x_test , y_train)
    logistic_regression_accuracy, svm_accuracy, DecisionTree_accuracy, cm_log, cm_svm, cm_decision = Model_Evaluation.model_evaluation(y_pred_log, y_pred_svm, y_pred_decision, y_test)
    
    #print("Logistic Regression Prediction : \n", y_pred_log, end ='\n')
    #print("SVM Prediction : \n", y_pred_svm, end = '\n')
    #print("Decision Tree Prediction : \n", y_pred_decision, end ='\n')
	
	
    print("Logistic Regression Accuracy : ",logistic_regression_accuracy, end = '\n')
    print("Confusion Matrix for Logistic Regression : \n", cm_log, end = '\n')
    print("SVM Accuracy : ", svm_accuracy, end = '\n')
    print("Confusion Matrix for SVM : \n", cm_svm, end = '\n')
    print("Decision Tree Accuracy : ", DecisionTree_accuracy, end = '\n')
    print("Confusion Matrix for DecisionTree : \n", cm_decision, end = '\n')
    
if __name__ == '__main__':
    main()
    
	
	
