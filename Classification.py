# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.ensemble import VotingClassifier

class Classification_:
    
    def classification(self, x_train , x_test , y_train):
        # Fitting Logistic Regression to the Training set
        classifier_log = LogisticRegression(random_state = 0)
        
        # Fitting SVM to the Training set
        classifier_svm = SVC(kernel = 'linear', random_state = 0)
        
        # Fitting DecisionTree to the Training set
        classifier_decision = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
        
        classifier_log.fit(x_train,y_train)
        classifier_svm.fit(x_train,y_train)
        classifier_decision.fit(x_train,y_train)
        
        # Voting Module
        vc = VotingClassifier([('log',classifier_log),('svm',classifier_svm),('tree',classifier_decision)],voting = 'hard')
        vc_clf = vc.fit(x_train,y_train)
        
        # Predicting the Test set results
        y_pred_log = classifier_log.predict(x_test)
        y_pred_svm = classifier_svm.predict(x_test)
        y_pred_decision = classifier_decision.predict(x_test)
        y_pred_vc = vc_clf.predict(x_test)
        
        return y_pred_log, y_pred_svm, y_pred_decision, vc_clf ,classifier_log , classifier_svm ,  classifier_decision ,  y_pred_vc
    
    def classifition_input(self ,x_input , classifier_log , classifier_svm ,  classifier_decision ,logistic_regression_accuracy, svm_accuracy, DecisionTree_accuracy,vc_accuracy , vc_clf) :
        x = max (svm_accuracy, logistic_regression_accuracy , vc_accuracy , DecisionTree_accuracy) 
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
            des_pre = classifier_decision.predict(x_input)
            return  des_pre