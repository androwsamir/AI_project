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
    
    x_train, x_test, y_train, y_test, x_val, y_val = Preprocess.preprocessing()
    y_pred_log, y_pred_svm, y_pred_decision, vc_clf ,classifier_log , classifier_svm ,  classifier_decision ,  y_pred_vc = Classification_1.classification(x_train , x_test , y_train, x_val, y_val)
    logistic_regression_accuracy, svm_accuracy, DecisionTree_accuracy, cm_log, cm_svm, cm_decision , vc_accuracy = Model_Evaluation_1.model_evaluation( y_pred_log, y_pred_svm, y_pred_decision, y_test , y_pred_vc)
    
    print("Logistic Regression Accuracy : ",logistic_regression_accuracy, end = '\n')
    print("Confusion Matrix for Logistic Regression : \n", cm_log, end = '\n')
    print("SVM Accuracy : ", svm_accuracy, end = '\n')
    print("Confusion Matrix for SVM : \n", cm_svm, end = '\n')
    print("Decision Tree Accuracy : ", DecisionTree_accuracy, end = '\n')
    print("Confusion Matrix for DecisionTree : \n", cm_decision, end = '\n')
    
    filename = input()
    filename.encode('utf-8').strip()
    x_input = Preprocess.preprocess_input(filename)
    predict_ = Classification_1.classifition_input(x_input, classifier_log, classifier_svm, classifier_decision, logistic_regression_accuracy, svm_accuracy, DecisionTree_accuracy, vc_accuracy, vc_clf)
    list1 = []
    for ele in predict_ :
        print(ele)
# =============================================================================
#     for ele in predict_ :
#         if(ele == '0'):
#             list1.append('B')
#         else:
#             list1.append('M')
#     
#     list2 = ([list1])
#     print(list2)
# =============================================================================
# =============================================================================
#     list1 = []
#     for element in range (30):
#         ele = float(input())+xzcxzc V
#         list1.append(ele)
#     list2 = ([list1])
# =============================================================================
# =============================================================================
#     sc = StandardScaler()
#     list2 = sc.fit_transform(list2)
#     if vc_clf.predict(list2) == 0 :
#         print("B")
#     else :
#         print("M")
#     
# =============================================================================
# =============================================================================
#     if z_pred_log == 0 :
#         print ("Logistic Regression predict : B\n")
#     else:
#         print("Logistic Regression predict : M\n")
#         
#     if z_pred_svm == 0 :
#         print ("SVM predict : B\n")
#     else:
#         print("SVM predict : M\n")
#         
#     if z_pred_decision == 0 :
#         print ("Decision Tree predict : B\n")
#     else:
#         print("Decision Tree predict : M\n")
#         
# =============================================================================
# =============================================================================
#     if z_pred_log == z_pred_svm
#         if z_pred_log == 0 :
#             print ("Logistic Regression predict : B\n")
#         else:
#             print("Logistic Regression predict : M\n")
#     
#     elif z_pred_log == z_pred_decision:
#         if z_pred_log == 0 :
#             print ("Logistic Regression predict : B\n")
#         else:
#             print("Logistic Regression predict : M\n")
#             
#     elif z_pred_svm == z_pred_decision :
#         if z_pred_svm == 0 :
#             print ("SVM predict : B\n")
#         else:
#             print("SVM predict : M\n")
#                    
#     print (z_pred_log, end ='\n')
#     print (z_pred_svm, end ='\n')
#     print (z_pred_decision, end ='\n')
#     
#     print("Logistic Regression Accuracy : ",logistic_regression_accuracy, end = '\n')
#     print("Confusion Matrix for Logistic Regression : \n", cm_log, end = '\n')
#     print("SVM Accuracy : ", svm_accuracy, end = '\n')
#     print("Confusion Matrix for SVM : \n", cm_svm, end = '\n')
#     print("Decision Tree Accuracy : ", DecisionTree_accuracy, end = '\n')
#     print("Confusion Matrix for DecisionTree : \n", cm_decision, end = '\n')
# =============================================================================
if __name__ == '__main__':
    main()
    
	
	
