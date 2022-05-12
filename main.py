# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import Preprocessing
import Classification
import Model_Evaluation
from sklearn.preprocessing  import StandardScaler
from sklearn.ensemble import VotingClassifier


def main():    
    x_train , x_test , y_train , y_test = Preprocessing.preprocessing()
    y_pred_log, y_pred_svm, y_pred_decision, vc_clf = Classification.classification(x_train , x_test , y_train)
    logistic_regression_accuracy, svm_accuracy, DecisionTree_accuracy, cm_log, cm_svm, cm_decision = Model_Evaluation.model_evaluation(y_pred_log, y_pred_svm, y_pred_decision, y_test)
    
    #print("Logistic Regression Prediction : \n", y_pred_log, end ='\n')
    #print("SVM Prediction : \n", y_pred_svm, end = '\n')
    #print("Decision Tree Prediction : \n", y_pred_decision, end ='\n')

# 0.004551	12.25	22.44	78.18	0.01608	0.08192	0.1256	0.052	0.01714	0.00941	0.01261	
# 0.01205	0.08203	0.1544	0.05976	622.9	0.2239	0.31	0.1804	1.139	1.577	
# 18.04	0.005096	0.123	0.002399	14.17	31.99	92.74	466.5	0.06335

    list1 = []
    for element in range (30):
        ele = float(input())
        list1.append(ele)
    list2 = ([list1])
    
    
    sc = StandardScaler()
    list2 = sc.fit_transform(list2)
    if vc_clf.predict(list2) == 0 :
        print("B")
    else :
        print("M")
    
    
        
    #0.01024 11.13 22.44 71.49 0.02912 0.09566 0.1087 0.08194 0.04824 0.03445	
    #0.02257 0.03051 0.08032 0.203 0.06552 436.6 0.28 0.3169 0.1782 1.467	
    #1.994	17.85	0.003495	0.1564	0.004723	12.02	28.26	77.8	378.4	0.06413

    
    
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
    # if z_pred_log == z_pred_svm
    #     if z_pred_log == 0 :
    #         print ("Logistic Regression predict : B\n")
    #     else:
    #         print("Logistic Regression predict : M\n")
    
    # elif z_pred_log == z_pred_decision:
    #     if z_pred_log == 0 :
    #         print ("Logistic Regression predict : B\n")
    #     else:
    #         print("Logistic Regression predict : M\n")
            
    # elif z_pred_svm == z_pred_decision :
    #     if z_pred_svm == 0 :
    #         print ("SVM predict : B\n")
    #     else:
    #         print("SVM predict : M\n")
            
    #print (z_pred_log, end ='\n')
    #print (z_pred_svm, end ='\n')
    #print (z_pred_decision, end ='\n')
    
    #print("Logistic Regression Accuracy : ",logistic_regression_accuracy, end = '\n')
    #print("Confusion Matrix for Logistic Regression : \n", cm_log, end = '\n')
    #print("SVM Accuracy : ", svm_accuracy, end = '\n')
    #print("Confusion Matrix for SVM : \n", cm_svm, end = '\n')
    #print("Decision Tree Accuracy : ", DecisionTree_accuracy, end = '\n')
    #print("Confusion Matrix for DecisionTree : \n", cm_decision, end = '\n')
    
if __name__ == '__main__':
    main()
    
	
	
