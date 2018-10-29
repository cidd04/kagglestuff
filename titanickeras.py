import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout


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


def split_valid_test_data(data, fraction=(1 - 0.8)):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)
    data_x = data.drop(["Survived"], axis=1)
    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)
    return train_x.values, train_y, valid_x, valid_y


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

train_x, train_y, valid_x, valid_y = split_valid_test_data(train_data)
print("train_x:{}".format(train_x.shape))
print("train_y:{}".format(train_y.shape))
print("train_y content:{}".format(train_y[:3]))

# print("valid_x:{}".format(valid_x.shape))
# print("valid_y:{}".format(valid_y.shape))

model = Sequential()
model.add(Dense(200))
model.add(Dropout(.25))
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Dropout(.25))
model.add(Activation('relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=500, verbose=2, batch_size=16)
