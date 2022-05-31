# Model Evaluating
from sklearn.metrics import confusion_matrix
from sklearn import metrics

class Model_Evaluation_:
    
    def model_evaluation(self,  y_pred_log, y_pred_svm, y_pred_naive, y_test , y_pred_vc):
        # Making the Confusion Matrix for Logistic Regression
        cm_log = confusion_matrix(y_test, y_pred_log)
        
        # Making the Confusion Matrix for SVM
        cm_svm = confusion_matrix(y_test, y_pred_svm)
        
        # Making the Confusion Matrix for DecisionTree
        cm_naive = confusion_matrix(y_test, y_pred_naive)
        
        #Model Accuracy : how often is the classifier correct?
        #Model Accuracy for Logistic Regression
        logistic_regression_accuracy = metrics.accuracy_score(y_test, y_pred_log)
        
        #Model Accuracy for SVM
        svm_accuracy = metrics.accuracy_score(y_test, y_pred_svm)
        
        #Model Accuracy for DecisionTree
        NaiveBayse_accuracy = metrics.accuracy_score(y_test, y_pred_naive)
        
        #Model Accuracy for  Voting 
        vc_accuracy = metrics.accuracy_score(y_test , y_pred_vc)
        
        return logistic_regression_accuracy, svm_accuracy, NaiveBayse_accuracy, cm_log, cm_svm, cm_naive , vc_accuracy   
