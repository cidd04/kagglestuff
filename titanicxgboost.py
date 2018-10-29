import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


def nan_padding(data, columns):
    for column in columns:
        imputer = SimpleImputer()
        data[column] = imputer.fit_transform(data[column].values.reshape(-1, 1))
    return data


def drop_not_concerned(data, columns):
    return data.drop(columns, axis=1)


def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data


def sex_to_int(data):
    le = LabelEncoder()
    le.fit(["male", "female"])
    data["Sex"] = le.transform(data["Sex"])
    return data


def normalize_age(data):
    scaler = MinMaxScaler()
    data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1, 1))
    return data


def split_valid_test_data(data):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)
    data_x = data.drop(["Survived"], axis=1)
    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y,
                                                          test_size=.3,
                                                          random_state=5,
                                                          stratify=data_y)
    return train_x.values, valid_x.values, train_y.ravel(), valid_y.ravel()


train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

nan_columns = ["Age", "SibSp", "Parch"]

train_data = nan_padding(train_data, nan_columns)
test_data = nan_padding(test_data, nan_columns)

test_passenger_id = test_data["PassengerId"]

not_concerned_columns = ["PassengerId", "Name", "Ticket", "Fare", "Cabin", "Embarked"]
train_data = drop_not_concerned(train_data, not_concerned_columns)
test_data = drop_not_concerned(test_data, not_concerned_columns)

print(train_data.head())

dummy_columns = ["Pclass"]
train_data = dummy_data(train_data, dummy_columns)
test_data = dummy_data(test_data, dummy_columns)

print(train_data.head())

train_data = sex_to_int(train_data)
test_data = sex_to_int(test_data)

print(train_data.head())

train_data = normalize_age(train_data)
test_data = normalize_age(test_data)

print(train_data.head())

train_x, valid_x, train_y, valid_y = split_valid_test_data(train_data)
print("train_x:{}".format(train_x.shape))
print("train_y:{}".format(train_y.shape))
print("train_y content:{}".format(train_y[:3]))

rf = RandomForestClassifier()

rf.fit(train_x, train_y)

print(rf.score(valid_x, valid_y))

xgb = XGBClassifier()
xgb.fit(train_x, train_y)
print(xgb.score(valid_x, valid_y))

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(train_x, train_y)
print(lg.score(valid_x, valid_y))

from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'n_estimators': range(8, 20),
    'max_depth': range(6, 10),
    'learning_rate': [.4, .45, .5, .55, .6],
    'colsample_bytree': [.6, .7, .8, .9, 1]
}

# Instantiate the regressor: gbm
gbm = XGBClassifier(n_estimators=10)

# Perform random search: grid_mse
xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid,
                                    estimator = gbm, scoring = "accuracy",
                                    verbose = 1, n_iter = 50, cv = 4)


# Fit randomized_mse to the data
xgb_random.fit(train_x, train_y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", xgb_random.best_params_)
print("Best accuracy found: ", xgb_random.best_score_)
