from flask import Flask, render_template,request,session,flash
import sqlite3 as sql
import os
import pandas as pd
import json


from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, LSTM
import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np  # linear algebra
#import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from easygui import *
        # %%
        # from keras.models import Sequential
        # from keras.layers import Dense, LSTM
import tensorflow as tf
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/gohome')
def homepage():
    return render_template('index.html')







@app.route('/enternew')
def new_user():
   return render_template('signup.html')

@app.route('/addrec',methods = ['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            nm = request.form['Name']
            phonno = request.form['MobileNumber']
            email = request.form['email']
            unm = request.form['Username']
            passwd = request.form['password']
            with sql.connect("agricultureuser.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO agriuser(name,phono,email,username,password)VALUES(?, ?, ?, ?,?)",(nm,phonno,email,unm,passwd))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"

        finally:
            return render_template("result.html", msg=msg)
            con.close()

@app.route('/userlogin')
def user_login():
   return render_template("login.html")

@app.route('/logindetails',methods = ['POST', 'GET'])
def logindetails():
    if request.method=='POST':
            usrname=request.form['username']
            passwd = request.form['password']

            with sql.connect("agricultureuser.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username,password FROM agriuser where username=? ",(usrname,))
                account = cur.fetchall()

                for row in account:
                    database_user = row[0]
                    database_password = row[1]
                    if database_user == usrname and database_password==passwd:
                        session['logged_in'] = True
                        return render_template('home.html')
                    else:
                        flash("Invalid user credentials")
                        return render_template('login.html')

@app.route('/predictinfo')
def predictin():
   return render_template('info.html')



@app.route('/predict',methods = ['POST', 'GET'])
def predcrop():
    if request.method == 'POST':
        comment = request.form['comment']
        comment1 = request.form['comment1']

        data = comment
        data1 = comment1

        # type(data2)
        print(data)
        print(data1)



        # Input data files are available in the read-only "../input/" directory
        # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

        import os
        dataframes = []
        stock_name = []

        data22 = data + '.csv'
        data33 = 'C:/Users/S Kiran/Programs Kiran/Stock_market_predict/stock_market_prediction/To_send/Data1/' + data22
        dataframes = pd.read_csv(data33)
        stock_name = data
        dataframes['Adj Close'].plot()
        plt.ylabel('Adj Close')
        plt.xlabel(None)
        plt.title(f"Closing Price of {stock_name[0]}")

        # %%
        dataframes['Volume'].plot()
        plt.ylabel('Volume')
        plt.xlabel(None)
        plt.title(f"Volume of {stock_name[0]}")
        plt.savefig("C:/Users/S Kiran/Programs Kiran/Stock_market_predict/stock_market_prediction/To_send/Flask_main/templates/graph3.png")

        plt.show()

        # %%
        ma_day = [10, 50, 100, 365]
        for ma in ma_day:
            # for company in dataframes:
            column_name = f"MA for {ma} days"
            print('column_name', column_name)
            # print('company', company)
            # rolling calculates mean over ma days in time series
            dataframes[column_name] = dataframes['Adj Close'].rolling(ma).mean()

        # %%
        #fig, axes = plt.subplots(nrows=2, ncols=5)
        #fig.set_figheight(8)
        #fig.set_figwidth(30)
        k = 0;
        j = 0;
        # for i in range(0,1):
        #dataframes[['Adj Close', 'MA for 10 days', 'MA for 50 days', 'MA for 100 days', 'MA for 365 days']].plot(
         #   ax=axes[k, j]).set_title(f"{stock_name}");
        j = j + 1;
        if (j == 5):
            k = 1;
            j = 0;
        #fig.tight_layout()
        # linestyle = '--', marker = 'o'
        print('reach1')
        # %%
        """
        ### Stock Returns:
        """

        # %%
        # for company in dataframes:
        dataframes['Daily Return'] = dataframes['Adj Close'].pct_change()
        # pct_change finds percent change.

        # %%
       # fig, axes = plt.subplots(nrows=2, ncols=5)
        #fig.set_figheight(8)
        #fig.set_figwidth(30)
        k = 0;
        j = 0;
        # for i in range(0,1):
        #dataframes[['Daily Return']].plot(ax=axes[k, j], linestyle='--', marker='o').set_title(f"{stock_name}");
        j = j + 1;
        if (j == 5):
            k = 1;
            j = 0;
        #fig.tight_layout()
        print('reach2')
        # %%
        """
        ### Build LSTM Models:
        """


        # %%
        def build_training_dataset(input_ds):
            # Create a new dataframe with only the 'Close column
            input_ds.reset_index()
            data = input_ds.filter(items=['Close'])
            # Convert the dataframe to a numpy array
            dataset = data.values
            # Get the number of rows to train the model on
            training_data_len = int(np.ceil(len(dataset) * .95))
            return data, dataset, training_data_len

        # Test the function
        training_data_df, training_dataset_np, training_data_len = build_training_dataset(dataframes)
        dataset = training_dataset_np
        data = training_data_df

        # %%

        def scale_the_data(dataset):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)
            return scaler, scaled_data

        # Test the function
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
                x_train.append(train_data[i - 60:i, 0])
                y_train.append(train_data[i, 0])
                if i <= 61:
                    # print(x_train)
                    # print(y_train)
                    print('.')

            # Convert to numpy arrays
            x_train, y_train = np.array(x_train), np.array(y_train)

            # Reshape the data
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            # x_train.shape
            return x_train, y_train

        # Test the function
        x_train, y_train = split_train_dataset(training_data_len)

        # %%
        def build_lstm_model(x_train, y_train):
            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(64, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            # Compile the model
            # adam ~ Stochastic Gradient descent method.
            model.compile(optimizer='adam', loss='mean_squared_error')
            # Train the model
            model.fit(x_train, y_train, batch_size=1, epochs=1)
            return model

            # Test the function

        lstm_model = build_lstm_model(x_train, y_train)
        print('reach3')

        # %%
        # import pickle
        # pickle.dump(lstm_model,open(stock_name[100],'wb'))
        lstm_model.save('model.h5')

        # %%
        def create_testing_data_set(model, scaler, training_data_len, test_data_len):
            # Create the testing data set
            # Create a new array containing scaled values from index 1543 to 2002
            test_data = scaled_data[training_data_len - test_data_len:, :]
            # Create the data sets x_test and y_test
            x_test = []
            y_test = dataset[training_data_len:, :]
            for i in range(test_data_len, len(test_data)):
                x_test.append(test_data[i - test_data_len:i, 0])

            # Convert the data to a numpy array
            x_test = np.array(x_test)

            # Reshape the data
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            # Get the models predicted price values
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            predicted_main_out = np.mean(predictions)

            # Get the root mean squared error (RMSE)
            rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
            rmse
            return (x_test, y_test, predictions, rmse, predicted_main_out)

        # Test the function
        TEST_DATA_LENGTH = 60
        print('reach4')
        x_test, y_test, predictions, rmse, predicted_main_out = create_testing_data_set(lstm_model, scaler, training_data_len,
                                                                    TEST_DATA_LENGTH)



        # %%
        def plot_predictions(stock, data, training_data_len):
            # Plot the data
            print('stockname', stock)
            train = data[:training_data_len]
            valid = data[training_data_len:]
            valid['Predictions'] = predictions
            # Visualize the data
            plt.figure(figsize=(16, 6))
            title = stock + ' Model Forecast'
            ylabel = stock + ' Close Price'
            plt.title(title)
            plt.xlabel('Date', fontsize=18)
            plt.ylabel(ylabel, fontsize=18)
            plt.plot(train['Close'])
            plt.plot(valid[['Close', 'Predictions']])
            plt.legend(['Training Data', 'Validated Data', 'Predicted Data'], loc='lower right')
            plt.savefig("C:/Users/S Kiran/Programs Kiran/Stock_market_predict/stock_market_prediction/To_send/Flask_main/templates/graph1.png")

            #plt.show()

            return valid

        # Test the function
        valid = plot_predictions('AMBIKCO.NS', data, training_data_len)
        print('valid',valid)


        # title of our window
        title = "The Predicted price"

        # message for our window
        msg = predicted_main_out

        # button message by default it is "OK"
        button = "Close"

        # creating a message box
        #msgbox(msg, title, button)
        response = predicted_main_out

        predicted_main_out = predicted_main_out.astype(float)

        # Create a dictionary to store key-value pairs
        data_dict = {"predicted_main_out": predicted_main_out.tolist()}
        #to store the predicted price in a json file
        with open('predicted_main_out.json', 'w') as f:
            json.dump(data_dict, f)

        
        return render_template('info.html',predicted_main_out=predicted_main_out)
        #return render_template('resultpred.html', prediction=response, price=statistics.mean(Price_Crop88), prediction1=response2, price1=statistics.mean(Price_Crop99),
                               #yeild88=statistics.mean(Yield_Crop88), yeild99 = statistics.mean(Yield_Crop99),
                               #prediction2=pred[0], price2=statistics.mean(Price_Crop1), prediction3=pred[1],
                               #price3=statistics.mean(Price_Crop2),yeild1 = statistics.mean(Yield_Crop1), yeild2 = statistics.mean(Yield_Crop2),
                               #prediction4=pred1[0], price4=statistics.mean(Price_Crop4), yeild4 = statistics.mean(Yield_Crop4),
                               #yeild5=statistics.mean(Yield_Crop5), prediction5=pred1[1],
                               #price5=statistics.mean(Price_Crop5))


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('login.html')

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)

