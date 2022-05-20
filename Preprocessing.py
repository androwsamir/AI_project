# Data Preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from pandas import DataFrame

class preprocess:
    
    def preprocessing(self):
        # Importing the dataset
        dataset = pd.read_csv('Tumor Cancer Prediction_Data.csv')
        x = dataset.iloc[:, 1:-1].values
        y = dataset.iloc[:, -1]

        # Encoding categorical data
        labelencoder_y = LabelEncoder()
        y = labelencoder_y.fit_transform(y)

        # Splitting the d ataset into the Training set and Test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

        # Feature Scalling
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.transform(x_test)
        return x_train, x_test, y_train, y_test

    def preprocess_input(self, filename):
        dataset = pd.read_csv(filename)
        dataset = dataset.dropna()
        dataset.drop_duplicates() 
        x_input = dataset.iloc[:,:].values
        sc_x = StandardScaler()
        x_input = sc_x.fit_transform(x_input)
       
        return x_input