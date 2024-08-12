# Machine Learning Projects

## Part A: Signal Quality Prediction

### Domain: Electronics and Telecommunication

### Project Overview

A communications equipment manufacturing company wants to predict the signal quality of their equipment using various measurable signal parameters. The objective is to build a classifier that can determine the signal strength or quality based on the given parameters.

### Data Description

- **Parameters:** Various measurable signal parameters.
- **Signal_Quality:** Final signal strength or quality.

### Steps and Tasks

#### 1. Data Import and Understanding

1. **Read the Dataset**
    - Import the dataset `Signals.csv` as a DataFrame and load the required libraries.
    - ```python
      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      import seaborn as sns

      df = pd.read_csv('Signals.csv')
      ```

2. **Check for Missing Values**
    - Check for missing values in the dataset and print the percentage for each attribute.
    - ```python
      missing_values = df.isnull().mean() * 100
      print(missing_values)
      ```

3. **Check for Duplicate Records**
    - Identify and handle duplicate records in the dataset.
    - ```python
      duplicates = df.duplicated().sum()
      df = df.drop_duplicates()
      ```

4. **Visualize Target Variable**
    - Plot the distribution of the target variable `Signal_Quality`.
    - ```python
      sns.countplot(x='Signal_Quality', data=df)
      plt.title('Distribution of Signal Quality')
      plt.show()
      ```

5. **Initial Data Analysis Insights**
    - Provide insights based on the initial data analysis.

#### 2. Data Preprocessing

1. **Split Data into Features and Target**
    - Split the data into features (X) and target (Y).
    - ```python
      X = df.drop('Signal_Quality', axis=1)
      Y = df['Signal_Quality']
      ```

2. **Train-Test Split**
    - Split the data into training and testing sets with a 70:30 proportion.
    - ```python
      from sklearn.model_selection import train_test_split
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
      ```

3. **Print Shape of Data**
    - Print the shape of all four variables and verify if the train and test data are in sync.
    - ```python
      print(f"X_train shape: {X_train.shape}")
      print(f"X_test shape: {X_test.shape}")
      print(f"Y_train shape: {Y_train.shape}")
      print(f"Y_test shape: {Y_test.shape}")
      ```

4. **Normalize Data**
    - Normalize the train and test data using an appropriate method.
    - ```python
      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_test = scaler.transform(X_test)
      ```

5. **Transform Labels**
    - Transform labels into a format acceptable by the Neural Network.
    - ```python
      from sklearn.preprocessing import LabelEncoder
      encoder = LabelEncoder()
      Y_train = encoder.fit_transform(Y_train)
      Y_test = encoder.transform(Y_test)
      ```

#### 3. Model Training & Evaluation using Neural Network

1. **Design Neural Network**
    - Design a Neural Network model to train the classifier.
    - ```python
      from tensorflow.keras.models import Sequential
      from tensorflow.keras.layers import Dense

      model = Sequential([
          Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
          Dense(32, activation='relu'),
          Dense(1, activation='sigmoid')
      ])
      ```

2. **Train the Model**
    - Train the classifier using the previously designed architecture.
    - ```python
      model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
      history = model.fit(X_train, Y_train, epochs=20, validation_split=0.2, batch_size=32)
      ```

3. **Plot Visuals**
    - Plot training and validation loss, and accuracy.
    - ```python
      # Loss plot
      plt.plot(history.history['loss'], label='Training Loss')
      plt.plot(history.history['val_loss'], label='Validation Loss')
      plt.title('Training and Validation Loss')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
      plt.show()

      # Accuracy plot
      plt.plot(history.history['accuracy'], label='Training Accuracy')
      plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('Epochs')
      plt.ylabel('Accuracy')
      plt.legend()
      plt.show()
      ```

4. **Update Architecture**
    - Update or redesign the architecture to improve performance and retrain the model.

5. **Compare Models**
    - Compare the updated model’s performance with the initial model and share insights.

## Part B: Digit Recognition from SVHN Dataset

### Domain: Autonomous Vehicles

### Project Overview

Recognizing multi-digit numbers from street-level photographs is crucial for map making. This project involves building a digit classifier using the SVHN (Street View House Number) dataset.

### Data Description

- **Dataset:** SVHN dataset consisting of images of house numbers from Google Street View.

### Steps and Tasks

#### 1. Data Import and Exploration

1. **Read the .h5 File**
    - Load the dataset from the `.h5` file.
    - ```python
      import h5py

      with h5py.File('svhn_dataset.h5', 'r') as f:
          # Print keys to explore the dataset
          print(list(f.keys()))
      ```

2. **Print Keys from .h5 File**
    - Print all the keys from the `.h5` file.
    - ```python
      print(list(f.keys()))
      ```

3. **Split Data**
    - Split the dataset into `X_train`, `X_test`, `Y_train`, `Y_test`.
    - ```python
      X_train = f['X_train'][:]
      X_test = f['X_test'][:]
      Y_train = f['Y_train'][:]
      Y_test = f['Y_test'][:]
      ```

#### 2. Data Visualisation and Preprocessing

1. **Print Shapes**
    - Print the shape of all split data to ensure consistency.
    - ```python
      print(f"X_train shape: {X_train.shape}")
      print(f"X_test shape: {X_test.shape}")
      print(f"Y_train shape: {Y_train.shape}")
      print(f"Y_test shape: {Y_test.shape}")
      ```

2. **Visualise Images**
    - Visualize the first 10 images in the training data and print their corresponding labels.
    - ```python
      fig, axes = plt.subplots(1, 10, figsize=(15, 5))
      for i in range(10):
          axes[i].imshow(X_train[i])
          axes[i].set_title(f'Label: {Y_train[i]}')
          axes[i].axis('off')
      plt.show()
      ```

3. **Reshape Images**
    - Reshape the images as needed.
    - ```python
      X_train = X_train.reshape(-1, 32, 32, 3)
      X_test = X_test.reshape(-1, 32, 32, 3)
      ```

4. **Normalize Images**
    - Normalize pixel values to [0, 1] range.
    - ```python
      X_train = X_train / 255.0
      X_test = X_test / 255.0
      ```

5. **Transform Labels**
    - Convert labels into a format acceptable by the Neural Network.
    - ```python
      from tensorflow.keras.utils import to_categorical
      Y_train = to_categorical(Y_train, num_classes=10)
      Y_test = to_categorical(Y_test, num_classes=10)
      ```

6. **Print Number of Classes**
    - Print the total number of classes in the dataset.
    - ```python
      print(f"Number of classes: {len(np.unique(Y_train))}")
      ```

#### 3. Model Training & Evaluation using Neural Network

1. **Design Neural Network**
    - Design a Neural Network model for digit classification.
    - ```python
      model = Sequential([
          Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
          MaxPooling2D((2, 2)),
          Conv2D(64, (3, 3), activation='relu'),
          MaxPooling2D((2, 2)),
          Conv2D(128, (3, 3), activation='relu'),
          Flatten(),
          Dense(128, activation='relu'),
          Dense(10, activation='softmax')
      ])
      ```

2. **Train the Model**
    - Train the model using the previously designed architecture.
    - ```python
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      history = model.fit(X_train, Y_train, epochs=10, validation_split=0.2, batch_size=32)
      ```

3. **Evaluate Model Performance**
    - Evaluate the model using appropriate metrics.
    - ```
