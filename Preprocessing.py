# Data Preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocessing():
    # Importing the dataset
    dataset = pd.read_csv('Tumor Cancer Prediction_Data.csv')
    x = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1]

    # Encoding categorical data
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    # Splitting the d ataset into the Training set and Test set

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=114)

    # Feature Scalling
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)

    return x_train, x_test, y_train, y_test
