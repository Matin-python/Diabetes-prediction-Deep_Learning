# 🩺 Diabetes Prediction using Deep Learning

A deep learning classification project that uses an Artificial Neural Network (ANN) to predict whether a patient has diabetes based on medical diagnostic measurements.

## Overview

This project uses the Diabetes dataset to build a binary classification model using Deep Learning.

The target variable, `Outcome`, indicates whether a patient has diabetes:

* `0` → No diabetes
* `1` → Diabetes

An Artificial Neural Network (ANN) is built using TensorFlow/Keras and trained to learn the relationship between the medical features and the diabetes outcome.

The project also includes data preprocessing, train/test splitting, model training, prediction, and evaluation.

## Features

* 🩺 Diabetes prediction using Deep Learning
* 🤖 Binary classification
* 🧠 Artificial Neural Network (ANN)
* 📊 Train/test data splitting
* 📈 Model training and evaluation
* 📉 Mean Squared Error (MSE) calculation
* 🐼 Dataset handling using Pandas
* 🔥 Neural network model built with TensorFlow/Keras
* 📚 Beginner-friendly Deep Learning implementation

## Technologies Used

* Python 3
* NumPy
* Pandas
* TensorFlow
* Keras
* Scikit-learn
* Matplotlib

## Dataset

The project uses a diabetes dataset containing medical diagnostic measurements used to predict whether a patient has diabetes.

The dataset contains the following features:

* `Pregnancies`
* `Glucose`
* `BloodPressure`
* `SkinThickness`
* `Insulin`
* `BMI`
* `DiabetesPedigreeFunction`
* `Age`

The target variable is:

* `Outcome`

where:

```text
0 = No Diabetes
1 = Diabetes
```

## Data Preparation

The dataset is loaded using Pandas.

The input features are separated from the target variable:

```python
X = df.drop('Outcome', axis=1)
y = df.Outcome
```

The dataset is then divided into training and testing sets using `train_test_split`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
```

The training data is used to train the neural network, while the test data is used to evaluate its performance on unseen samples.

## Model

The project uses an Artificial Neural Network built with TensorFlow/Keras.

The model contains a dense hidden layer with ReLU activation and an output layer for the binary classification task.

```python
model = Sequential()

model.add(Dense(20, input_dim=8, activation='relu'))
model.add(Dense(1))
```

The model is compiled using the Adam optimizer and Mean Squared Error loss:

```python
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mean_squared_error']
)
```

The model is then trained for multiple epochs:

```python
h = model.fit(
    X_train,
    y_train,
    epochs=200
)
```



## License

This project is licensed under the MIT License.

## Author

Mohammad Reza Bakhshandeh

Electrical Engineering (Electronics) Graduate

Interested in Python Development, Machine Learning, Deep Learning, Computer Vision, Artificial Intelligence, and Game Development.
