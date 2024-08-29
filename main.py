import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

print(tf.keras.__version__)

# 1. Laden des Datensatzes
df = pd.read_csv('train.csv')

# 2. Preprocessing
# Auswahl der relevanten Features und Zielvariable
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
target = 'Survived'

# Umgang mit fehlenden Werten
df = df.assign(Age=df['Age'].fillna(df['Age'].median()))
df = df.assign(Embarked=df['Embarked'].fillna(df['Embarked'].mode()[0]))

# One-Hot-Encoding für 'Sex' und 'Embarked'
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Trennung in Features und Zielvariable
X = df[features]
y = df[target]

# Normalisierung der numerischen Features
scaler = StandardScaler()
X.loc[:, ['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])

# Aufteilung des Datensatzes in Trainings- und Testdatensatz (90% Training, 10% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 3. Deep Learning Modell erstellen
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Modell kompilieren
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modell trainieren
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.1, verbose=1)

# 4. Evaluierung des Modells
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
