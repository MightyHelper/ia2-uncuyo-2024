from pathlib import Path

import keras
import pandas as pd
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LeakyReLU
from keras.src.metrics import AUC
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split

# import tensorflow

ROOT = Path(".")
TRAIN_CSV = ROOT / "train.csv"

train_set = pd.read_csv(TRAIN_CSV)


def preprocess(df):
    # Shuffle
    # One-hot: `Pclass`, `Sex`, `Embarked`
    # Na or ends in .5
    df["AgeIsMissing"] = (df["Age"].isna() | df["Age"] % 1 == 0.5)
    # Fill na: `Age`, `Fare`
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Cabin"] = df["Cabin"].fillna("Z_Unknown")
    df["TicketIsOnlyNumbers"] = df["Ticket"].str.isnumeric()
    df["TicketStringLength"] = df["Ticket"].str.len()
    df["TicketContainsSlash"] = df["Ticket"].str.contains("/")
    df["TicketContainsPeriod"] = df["Ticket"].str.contains(".")
    df["SpacesInCabin"] = df["Cabin"].str.count(" ")
    df["FirstLetterOfCabin"] = df["Cabin"].str[0]
    df["NameLength"] = df["Name"].str.len()
    df["NameParts"] = df["Name"].str.count(" ")
    df = pd.get_dummies(df, columns=["Pclass", "Sex", "Cabin", "Embarked", "SpacesInCabin", "FirstLetterOfCabin"])
    df = df.drop(columns=["Name", "Ticket", "PassengerId"])
    print(df.columns)
    return df


def normalize(row, mean, std):
    return (row - mean) / std


preprocessed_train = preprocess(train_set)
early_stopping = EarlyStopping(monitor='val_auc', patience=150, min_delta=0.001, verbose=1, mode='max',
                               restore_best_weights=True)
schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=70, verbose=1, mode='max', min_delta=0.0005)
model = keras.Sequential([
    keras.layers.Dense(2 ** 6, activation=LeakyReLU()),
    keras.layers.GaussianDropout(rate=0.5),
    keras.layers.Dense(2 ** 5, activation=LeakyReLU()),
    keras.layers.GaussianDropout(rate=0.5),
    keras.layers.Dense(2 ** 4, activation=LeakyReLU()),
    keras.layers.GaussianDropout(rate=0.5),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer=Adam(learning_rate=0.005), loss="binary_crossentropy", metrics=[AUC(), 'accuracy'])

X = preprocessed_train.drop(columns=["Survived"])
y = preprocessed_train["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123, stratify=y)

fare = X_train["Fare"].mean(), X_train["Fare"].std()
ticket_length = X_train["TicketStringLength"].mean(), X_train["TicketStringLength"].std()
name_length = X_train["NameLength"].mean(), X_train["NameLength"].std()
name_parts = X_train["NameParts"].mean(), X_train["NameParts"].std()

X_train["Fare"] = normalize(X_train["Fare"], *fare)
X_train["TicketStringLength"] = normalize(X_train["TicketStringLength"], *ticket_length)
X_train["NameLength"] = normalize(X_train["NameLength"], *name_length)
X_train["NameParts"] = normalize(X_train["NameParts"], *name_parts)
X_test["Fare"] = normalize(X_test["Fare"], *fare)
X_test["TicketStringLength"] = normalize(X_test["TicketStringLength"], *ticket_length)
X_test["NameLength"] = normalize(X_test["NameLength"], *name_length)
X_test["NameParts"] = normalize(X_test["NameParts"], *name_parts)

# X_train = X_train.sample(frac=5, random_state=123, replace=True)

history = model.fit(X_train, y_train, epochs=1000, batch_size=len(X_train), validation_data=(X_test, y_test),
                    callbacks=[early_stopping, schedule])

# Save model
model.save("./model.keras")

# Best validation score:
print(max(history.history["val_auc"]))
print(model.summary())
