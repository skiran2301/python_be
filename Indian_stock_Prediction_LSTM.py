
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
dataframes = []
stock_name = []
for dirname, _, filenames in os.walk('D:/Desktop/To_send/To_send/Data1'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        dataframes.append(pd.read_csv(os.path.join(dirname, filename),index_col = 0) )
        stock_name.append(filename)
        #print(dirname)
        #print(filename)
dataframes = pd.read_csv('D:/Desktop/To_send/To_send/Data1/AMBIKCO.NS.csv')
stock_name = 'AMBIKCO.NS'
dataframes['Adj Close'].plot()
plt.ylabel('Adj Close')
plt.xlabel(None)
plt.title(f"Closing Price of {stock_name[0]}")

# %%
dataframes['Volume'].plot()
plt.ylabel('Volume')
plt.xlabel(None)
plt.title(f"Volume of {stock_name[0]}")
plt.show()



# %%
ma_day = [10,50,100,365]
for ma in ma_day:
    #for company in dataframes:
    column_name = f"MA for {ma} days"
    print('column_name',column_name)
    #print('company', company)
    #rolling calculates mean over ma days in time series
    dataframes[column_name] = dataframes['Adj Close'].rolling(ma).mean()

# %%
fig, axes = plt.subplots(nrows=2, ncols=5)
fig.set_figheight(8)
fig.set_figwidth(30)
k = 0; j = 0;
#for i in range(0,1):
dataframes[['Adj Close', 'MA for 10 days', 'MA for 50 days', 'MA for 100 days','MA for 365 days']].plot(ax=axes[k,j]).set_title(f"{stock_name}");
j = j+1;
if(j==5):
    k=1;
    j=0;
fig.tight_layout()
# linestyle = '--', marker = 'o'

# %%
"""
### Stock Returns:
"""

# %%
#for company in dataframes:
dataframes['Daily Return'] = dataframes['Adj Close'].pct_change()
#pct_change finds percent change.

# %%
fig, axes = plt.subplots(nrows=2, ncols=5)
fig.set_figheight(8)
fig.set_figwidth(30)
k = 0; j = 0;
#for i in range(0,1):
dataframes[['Daily Return']].plot(ax=axes[k,j], linestyle = '--', marker = 'o').set_title(f"{stock_name}");
j = j+1;
if(j==5):
    k=1;
    j=0;
fig.tight_layout()

# %%
"""
### Build LSTM Models:
"""

from tensorflow import keras 
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, LSTM
# %%
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
import tensorflow as tf

# %%
def build_training_dataset(input_ds):
    # Create a new dataframe with only the 'Close column 
    input_ds.reset_index()
    data = input_ds.filter(items=['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .95 ))
    return data, dataset, training_data_len

#Test the function
training_data_df, training_dataset_np, training_data_len = build_training_dataset(dataframes)
dataset=training_dataset_np
data=training_data_df

# %%
from sklearn.preprocessing import MinMaxScaler
def scale_the_data(dataset):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    return scaler, scaled_data

#Test the function
scaler, scaled_data = scale_the_data(training_dataset_np)

# %%
# Create the training data set 
# Create the scaled training data set
def split_train_dataset(training_data_len):
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 61:
            #print(x_train)
            #print(y_train)
            print('.')
            
    # Convert to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_train.shape
    return x_train, y_train

#Test the function
x_train,y_train = split_train_dataset(training_data_len)

# %%
def build_lstm_model(x_train,y_train):
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    # adam ~ Stochastic Gradient descent method.
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model 

#Test the function
lstm_model = build_lstm_model(x_train,y_train)

# %%
#import pickle
#pickle.dump(lstm_model,open(stock_name[100],'wb'))
lstm_model.save('model.h5')

# %%
def create_testing_data_set(model, scaler, training_data_len,test_data_len):
    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002 
    test_data = scaled_data[training_data_len - test_data_len: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(test_data_len, len(test_data)):
        x_test.append(test_data[i-test_data_len:i, 0])
    
    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    rmse
    return (x_test, y_test, predictions, rmse)

#Test the function
TEST_DATA_LENGTH = 60
x_test,y_test, predictions, rmse = create_testing_data_set(lstm_model,scaler,training_data_len, TEST_DATA_LENGTH)

# %%
def plot_predictions(stock, data,training_data_len):
    #Plot the data
    print('stockname', stock)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16,6))
    title = stock + ' Model Forecast'
    ylabel = stock + ' Close Price'
    plt.title(title)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Training Data', 'Validated Data', 'Predicted Data'], loc='lower right')
    plt.show()
    return valid
    
#Test the function
valid = plot_predictions('AMBIKCO.NS.csv',data,training_data_len)
'''
# %%
from math import sqrt
from sklearn.metrics import mean_squared_error
i = 0
TEST_DATA_LENGTH = 111
error_scores = {}
trained_model = {}
print('stock_name',stock_name)

stock_name = 'AHLWEST.NS.csv'
for stock in stock_name:
    print('stock_name11', stock_name)
    df= dataframes[i]
    dataframes[i].dropna(inplace=True)
    #i= i+1
    training_data_df, training_dataset_np, training_data_len = build_training_dataset(df) #Build the Training Dataset
    dataset=training_dataset_np
    data=training_data_df
    scaler, scaled_data = scale_the_data(training_dataset_np) #Scale the data
    x_train,y_train = split_train_dataset(training_data_len) #split the data
    lstm_model = build_lstm_model(x_train,y_train) #build the LSTM model
    trained_model[stock] = lstm_model
    x_test,y_test, predictions, rmse = create_testing_data_set(lstm_model,scaler,training_data_len, TEST_DATA_LENGTH ) #create testing dataset and predictions
    valid = plot_predictions(stock,data,training_data_len) #plot predictions
    valid   # Show the valid and predicted prices
    rmse = sqrt(mean_squared_error(valid['Close'], valid['Predictions']))
    print('Test RMSE: %.3f' % (rmse))
    #error_scores.append(rmse)
    error_scores[stock] = rmse

print(error_scores)

# %%
'''