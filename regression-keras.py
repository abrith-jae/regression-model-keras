# Regression Model in Keras
# Author: Andrea Taylor
# Date: 12/02/2024

# Pre-Install needed libraries DO THIS PRIOR TO IMPORTING LIBRARIES
!pip install numpy==1.21.4
!pip install pandas==1.3.4
!pip install keras==2.1.6

# Import libraries
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter('ignore', FutureWarning)

# Download the data and read it into a pandas dataframe.
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()

# Let's check how many data points we have.
concrete_data.shape

# Let's check the dataset for any missing values.
concrete_data.describe()

concrete_data.isnull().sum()

# Split data into predictors and target
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

# Let's do a quick sanity check of the predictors and the target dataframes.
predictors.head()

target.head()

# Normalize the data by substracting the mean and dividing by the standard deviation.
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

# Build a Neural Network
# Import the Keras library
import keras

# Import the rest of the packages from the Keras library that is needed
from keras.models import Sequential
from keras.layers import Dense

# Define a function that defines the regression model to be able to conveniently call it to create the model.
# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and Test the Network
# Call the function to build the model
model = regression_model()

# Train and test the model at the same time using the fit method. Leave out 30% of the data for validation and train the model for 100 epochs.
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)
