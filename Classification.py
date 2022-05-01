import Preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def classification():
    x_train , x_test , y_train , y_test = Preprocessing.preprocessing()
    
    # Fitting Logistic Regression to the Training set
    classifier_log = LogisticRegression(random_state = 0)
    
    # Fitting SVM to the Training set
    classifier_svm = SVC(kernel = 'linear', random_state = 0)
    
    # Fitting DecisionTree to the Training set
    classifier_decision = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
    
    classifier_log.fit(x_train,y_train)
    classifier_svm.fit(x_train,y_train)
    classifier_decision.fit(x_train,y_train)
    
    # Predicting the Test set results
    y_pred_log = classifier_log.predict(x_test)
    y_pred_svm = classifier_svm.predict(x_test)
    y_pred_decision = classifier_decision.predict(x_test)
    
    return y_pred_log, y_pred_svm, y_pred_decision, y_test