# Data Preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def outliers (df, ft):
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound= Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    ls = df.index[(df[ft]<lower_bound)|(df[ft]>upper_bound)]
    
    return ls
    
def remove_outliers (df,ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    
    return df

class preprocess:
    
    def preprocessing(self):
        # Importing the dataset
        dataset1 = pd.read_csv('Tumor Cancer Prediction_Data.csv')
         
        index_list = []
        l = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13', 'F14', 'F15','F16', 'F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28','F29','F30']
        for feature in l :
            index_list.extend(outliers(dataset1,feature))
        dataset = remove_outliers(dataset1,index_list)
        
        dataset.drop('Index',axis = 'columns',inplace = True)
        
        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1]  
        
        # Encoding categorical data
        labelencoder_y = LabelEncoder()
        y = labelencoder_y.fit_transform(y)
        
        # Splitting the dataset into the Training set and Test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
        
        # Feature Scalling
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)   
        x_test = sc_x.transform(x_test)
        
        return x_train, x_test, y_train, y_test, sc_x
    
    
    def preprocess_input(self, filename, sc_x):
        filename.encode('utf-8').strip()
        dataset = pd.read_csv(filename)
        dataset.drop('Index',axis = 'columns',inplace = True)
        dataset = dataset.dropna()
        dataset.drop_duplicates()
        x_input = dataset.iloc[:,:-1].values
        x_input = sc_x.transform(x_input)
       
        return x_input