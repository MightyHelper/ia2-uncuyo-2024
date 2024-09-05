from pathlib import Path

import keras
import pandas as pd
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from tensorflow.python.keras.utils.version_utils import callbacks

# import tensorflow

ROOT = Path("/") / "mnt" / "e" / "datasets" / "titanic"
TRAIN_CSV = ROOT / "train.csv"

train_set = pd.read_csv(TRAIN_CSV)
def preprocess(df):
    # One-hot: `Pclass`, `Sex`, `Embarked`
    df = pd.get_dummies(df, columns=["Pclass", "Sex", "Embarked"])
    # Na or ends in .5
    df["AgeIsMissing"] = (df["Age"].isna() | df["Age"] % 1 == 0.5)
    # Fill na: `Age`, `Fare`
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    # Drop: `Name`, `Ticket`, `Cabin`
    df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])
    return df

preprocessed_train = preprocess(train_set)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, min_delta = 0.01, verbose = 1)


model = keras.Sequential([
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["F1Score"])

X = preprocessed_train.drop(columns=["Survived"])
y = preprocessed_train["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train, epochs=100, batch_size=32 * 23, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save model
model.save("./model.keras")
