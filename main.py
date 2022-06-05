# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import Preprocessing
import Classification
import Model_Evaluation
from sklearn.preprocessing  import StandardScaler
from sklearn.ensemble import VotingClassifier

    
def main():
    Preprocess = Preprocessing.preprocess()
    Classification_1 = Classification.Classification_()
    Model_Evaluation_1 = Model_Evaluation.Model_Evaluation_()
    
    x_train, x_test, y_train, y_test, sc_x = Preprocess.preprocessing()
    y_pred_log, y_pred_svm, y_pred_naive, vc_clf ,classifier_log , classifier_svm ,  classifier_naive ,  y_pred_vc = Classification_1.classification(x_train , x_test , y_train)
    logistic_regression_accuracy, svm_accuracy, NaiveBayse_accuracy, cm_log, cm_svm, cm_naive , vc_accuracy = Model_Evaluation_1.model_evaluation( y_pred_log, y_pred_svm, y_pred_naive, y_test , y_pred_vc)
    
    print("Logistic Regression Accuracy : ",logistic_regression_accuracy, end = '\n')
    print("Confusion Matrix for Logistic Regression : \n", cm_log, end = '\n')
    print("SVM Accuracy : ", svm_accuracy, end = '\n')
    print("Confusion Matrix for SVM : \n", cm_svm, end = '\n')
    print("Naive Bayse Accuracy : ", NaiveBayse_accuracy, end = '\n')
    print("Confusion Matrix for Naive Bayse : \n", cm_naive, end = '\n')
    print("Voting Module Accuracy : ",vc_accuracy , end = '\n')
    
    filename = input()
    x_input = Preprocess.preprocess_input(filename, sc_x)
    predict_ = Classification_1.classifition_input(x_input, classifier_log, classifier_svm, classifier_naive, logistic_regression_accuracy, svm_accuracy, NaiveBayse_accuracy, vc_accuracy, vc_clf)
    list1 = []
    for ele in predict_ :
        if(ele == 0):
            print('B')
        else :
            print('M')

if __name__ == '__main__':
    main()
    
	
	
