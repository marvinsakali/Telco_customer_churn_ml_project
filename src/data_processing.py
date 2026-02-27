"""
processing the data :
1. getting x and y
2. scaling the training datset to prevent overfiting the model
3. Standadizing the data to have the same range of values (0,1)

"""
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np

df = pd.read_csv(r'D:\Machine_Learning\Telco_churn\Data\processed\cleaned.csv')


train, val, test = np.split(df, [int(0.6 * len(df)), int( 0.8 * len(df))])
# function to get xy


def get_xy(dataframe, y_label, x_label=None,  oversample=False):
    dataframe = df.copy()

    if x_label is None:
        X = dataframe[[c for c in dataframe.columns if c != y_label]].values
    else:
        if len(x_label) == 1:
            X = dataframe[x_label[0]].values.reshape(-1, 1)
        else:
            X = dataframe[x_label].values
    y = dataframe[y_label].values.reshape(-1, 1)

    data = np.hstack([X, y])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample is True:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
    return data, X, y


x_columns = [c for c in df.columns if c != 'Churn']
_, X_train, y_train = get_xy(train, 'Churn', x_label=x_columns, oversample=True)
_, X_val, y_val = get_xy(val, 'Churn', x_label=x_columns, oversample=False)
_, X_test, y_test = get_xy(test, 'Churn', x_label=x_columns, oversample=True)


print(sum(y_train == 1))
print(sum(y_train == 0))
