from Data.processed.data_processing import get_xy, train_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


train, val, test = train_split(df)

_, X_train, y_train = get_xy(train, x_label=None, oversample= True)
_, X_val, y_val = get_xy(val, x_label=None, oversample= False)
_, X_test, y_test, y_test = get_xy(test, x_label=None, oversample= False)
# FUNCTION TO TRAIN MODEL
 
def train(model_class , X_train, y_train, **model_kwargs):
    model = model_class(**model_kwargs)
    model.fit(X_train, y_train)
    return model
 