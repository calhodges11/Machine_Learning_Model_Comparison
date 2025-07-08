import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#************************* SETUP ******************************
#Load the data for housing data from California and Iowa datasets. The Cali Data must be encoded due to some fields
#being left NaN
caliFilePath = "housing.csv"
iowaFilePath = "train.csv"
caliData = pd.read_csv(caliFilePath)
caliData_encoded = pd.get_dummies(caliData, columns=['ocean_proximity'])
iowaData = pd.read_csv(iowaFilePath)

# Set Y to sale price, set X as important house qualities
y = iowaData.SalePrice
#y = caliData.median_house_value
iowaQualities = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#X = caliData_encoded.drop('median_house_value', axis=1)
X = iowaData[iowaQualities]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#************** Random Forest ***********************
# Define a random forest model and train. We must use the imputer to replace NaN values with the mean.

# Replace NaN values with the mean
imputer = SimpleImputer(strategy='mean')
train_X = imputer.fit_transform(train_X)
val_X = imputer.transform(val_X)

startTimeRF = time.time()
rf_model = RandomForestRegressor()
rf_model.fit(train_X, train_y)

#Predict prices then calculate MAE and time.
predRF = rf_model.predict(val_X)
maeRF = mean_absolute_error(predRF, val_y)
endTimeRF = time.time()
timeRF = endTimeRF - startTimeRF

#***************** Linear Regression ***********************
# Initialize the linear regression model and train
startTimeLR = time.time()
# Create a Ridge regression model with regularization parameter alpha=3.0
regressor = Lasso(alpha=3.0)
regressor.fit(train_X, train_y)

# Make predictions on the test set, then calculate MAE and time elapsed
predLR = regressor.predict(val_X)
maeLR = mean_absolute_error(predLR, val_y)
endTimeLR = time.time()
timeLR = endTimeLR - startTimeLR

#******* Neural Network ***********
# Normalize the input data using StandardScaler
startTimeNN = time.time()
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
val_X = scaler.transform(val_X)

# Define the input shape
input_shape = (train_X.shape[1],)

# Initialize the model
model = Sequential()

# Add a dense layer with 100 neurons and ReLU activation function as the first hidden layer
model.add(Dense(100, input_shape=input_shape, activation='relu'))

# Add another dense layer with 1000 neurons and ReLU activation function as the second hidden layer
model.add(Dense(1000, activation='relu'))

# Add a dense layer with 1 neuron and linear activation function as the output layer
model.add(Dense(1, activation='relu'))

# Compile the model using mean absolute error as the loss function and Adam optimizer
model.compile(loss='mae', optimizer='adam')

# Train the model on the training data and print MAE of neural network
history = model.fit(train_X, train_y, epochs=10, batch_size=32, validation_data=(val_X, val_y))

# Predict the housing prices on the validation set
predNN = model.predict(val_X)
endTimeNN = time.time()
# Calculate the mean absolute error and runtime
maeNN = mean_absolute_error(val_y, predNN)
timeNN = endTimeNN - startTimeNN

#****************** Display Results **********************
#Print MAE and Time
print("MAE of Neural Network: {:.2f}".format(maeNN))
print("Elapsed time: {:,.2f} seconds".format(timeNN))
print("MAE of Random Forest Model: {:,.0f}".format(maeRF))
print("Elapsed time: {:,.0f} seconds".format(timeRF))
print("MAE of Linear Regression Model: {:,.0f}".format(maeLR))
print("Elapsed time: {:,.0f} seconds".format(timeLR))


# create scatterplots
plt.scatter(val_y, predLR, label='Linear Regression')
plt.scatter(val_y, predNN, label='Neural Network')
plt.scatter(val_y, predRF, label='Random Forest')
plt.scatter(val_y, val_y, label='Actual Prices', c='black')


plt.legend()
plt.show()






