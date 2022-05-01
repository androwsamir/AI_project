# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import Model_Evaluation

#Preprocessing.preprocessing()
def main():    
    logistic_regression_accuracy, svm_accuracy, DecisionTree_accuracy = Model_Evaluation.model_evaluation()
    print("Logistic Regression Accuracy : ",logistic_regression_accuracy, end = '\n')
    print("SVM Accuracy : ", svm_accuracy, end = '\n')
    print("Decision Tree Accuracy : ", DecisionTree_accuracy, end = '\n')
    
if __name__ == '__main__':
    main()