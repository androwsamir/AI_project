# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing  import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn import metrics

class Classification_:
    
    def classification(self, x_train , x_test , y_train):
        
        # Training Logistic Regression to the Training set
        classifier_log = LogisticRegression(C = 0.05)  # inverse of regularization strength
                                                       # smaller values specify stronger regularization
        # Training SVM to the Training set
        classifier_svm = SVC(kernel = 'linear', random_state = 0, C = 0.02)
        
        # Training Naive Bayse Classification model to the Training set
        classifier_naive = GaussianNB()
        
        classifier_log.fit(x_train,y_train)
        classifier_svm.fit(x_train,y_train)
        classifier_naive.fit(x_train,y_train)
        
        # Voting Module
        vc = VotingClassifier([('log',classifier_log),('svm',classifier_svm),('naive',classifier_naive)],voting = 'hard')
        vc_clf = vc.fit(x_train,y_train)
        
        # Predicting the Test set results
        y_pred_log = classifier_log.predict(x_test)
        y_pred_svm = classifier_svm.predict(x_test)
        y_pred_naive = classifier_naive.predict(x_test)
        y_pred_vc = vc_clf.predict(x_test)
        
        # Predicting the Training set results
        y_pred_log_train = classifier_log.predict(x_train)
        y_pred_svm_train = classifier_svm.predict(x_train)
        y_pred_naive_train = classifier_naive.predict(x_train)
        y_pred_vc_train = vc_clf.predict(x_train)
        
        #---------------------------------------------------------------#
        logistic_regression_accuracy_train = metrics.accuracy_score(y_train, y_pred_log_train)
        
        #Model Accuracy for SVM
        svm_accuracy_train = metrics.accuracy_score(y_train, y_pred_svm_train)
        
        #Model Accuracy for DecisionTree
        NaiveBayse_accuracy_train = metrics.accuracy_score(y_train, y_pred_naive_train)
        
        #Model Accuracy for  Voting 
        vc_accuracy_train = metrics.accuracy_score(y_train , y_pred_vc_train)
        
        print("Logistic Regression Accuracy Train : ",logistic_regression_accuracy_train, end = '\n')        
        print("SVM Accuracy Train : ", svm_accuracy_train, end = '\n')
        print("Naive Bayse Accuracy Train : ", NaiveBayse_accuracy_train, end = '\n')
        print("Voting Module Accuracy Train : ",vc_accuracy_train , end = '\n')
        
        return y_pred_log, y_pred_svm, y_pred_naive, vc_clf ,classifier_log , classifier_svm ,  classifier_naive ,  y_pred_vc
    
    def classifition_input(self ,x_input, classifier_log, classifier_svm, classifier_naive, logistic_regression_accuracy, svm_accuracy, NaiveBayse_accuracy, vc_accuracy, vc_clf) :
        x = max (svm_accuracy, logistic_regression_accuracy , vc_accuracy , NaiveBayse_accuracy) 
        if x == svm_accuracy :
            svm_pre = classifier_svm.predict(x_input)
            return svm_pre
        elif x == logistic_regression_accuracy :
            log_pre = classifier_log.predict(x_input)
            return log_pre
        elif x == vc_accuracy :
            vc_pre = vc_clf.predict(x_input)
            return vc_pre
        else :
            naive_pre = classifier_naive.predict(x_input)
            return  naive_pre