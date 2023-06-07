import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import plotly.graph_objs as go
import yfinance as yf

# Retrieve data for multiple stocks
df_axis = yf.Ticker("AXISBANK.NS").history(period='3y').reset_index()
df_sbi = yf.Ticker("SBIN.NS").history(period='3y').reset_index()
df_rbl = yf.Ticker("RBLBANK.NS").history(period='3y').reset_index()
df_pnb = yf.Ticker("PNB.NS").history(period='3y').reset_index()
df_kot = yf.Ticker("KOTAKBANK.NS").history(period='3y').reset_index()
df_ind = yf.Ticker("INDUSINDBK.NS").history(period='3y').reset_index()
df_idfc = yf.Ticker("IDFCFIRSTB.NS").history(period='3y').reset_index()
df_icic = yf.Ticker("ICICIBANK.NS").history(period='3y').reset_index()
df_band = yf.Ticker("BANDHANBNK.NS").history(period='3y').reset_index()
df_hdfc = yf.Ticker("HDFC.NS").history(period='3y').reset_index()
df_fed = yf.Ticker("FEDERALBNK.NS").history(period='3y').reset_index()
df_au = yf.Ticker("AUBANK.NS").history(period='3y').reset_index()

# Combine the dataframes and assign names
stocks = {
    'Axis Bank': df_axis,
    'SBI Bank': df_sbi,
    'RBL Bank': df_rbl,
    'PNB Bank': df_pnb,
    'Kotak Bank': df_kot,
    'IndusInd Bank': df_ind,
    'IDFC First Bank': df_idfc,
    'ICICI Bank': df_icic,
    'Bandhan Bank': df_band,
    'HDFC Bank': df_hdfc,
    'Federal Bank': df_fed,
    'AU Bank': df_au
}

Scale = StandardScaler()


def data_prep(df, lookback, future, Scale):
    date_train = pd.to_datetime(df['Date'])
    df_train = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
    df_train = df_train.astype(float)

    df_train_scaled = Scale.fit_transform(df_train)

    X, y = [], []
    for i in range(lookback, len(df_train_scaled) - future + 1):
        X.append(df_train_scaled[i - lookback:i, 0:df_train.shape[1]])
        y.append(df_train_scaled[i + future - 1:i + future, 0])

    return np.array(X), np.array(y), df_train, date_train


def Lstm_model1(X, y):
    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    regressor.fit(X, y, epochs=256, validation_split=0.1, batch_size=64, verbose=1, callbacks=[es])

    return regressor


app = Flask(__name__)

models = {}


@app.route("/")
def Home():
    return render_template("index.html", stocks=stocks)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict_open():
    stock_name = request.form.get('stock_name')
    model = models.get(stock_name)
    if model is None:
        return "Model not found for the selected stock."

    df = stocks.get(stock_name)
    if df is None:
        return "Data not found for the selected stock."

    lookback = 30
    future = 4
    X, y, df_train, date_train = data_prep(df, lookback, future, Scale)
    predicted = model.predict(X)
    predicted1 = np.repeat(predicted, df_train.shape[1], axis=-1)
    predicted_descaled = Scale.inverse_transform(predicted1)[:, 0]

    # Actual vs Predicted Prices plot
    fig_actual = go.Figure()
    fig_actual.add_trace(go.Scatter(x=date_train, y=df_train['Open'], name='Actual Closing Price', line=dict(color='blue')))
    fig_actual.add_trace(go.Scatter(x=date_train[lookback:], y=predicted_descaled, name='Predicted Closing Price', line=dict(color='red')))
    fig_actual.update_layout(title=f'{stock_name} - Actual vs Predicted Prices', xaxis_title='Date', yaxis_title='Close Price')
    plot_div_actual = fig_actual.to_html(full_html=False)

    # Predicted Future Prices plot
    future_dates = pd.date_range(date_train.iloc[-1], periods=future, freq='1d').tolist()
    future_dates = [date.date() for date in future_dates]

    predicted_future = pd.DataFrame({
        'Date': future_dates,
        'Open': predicted_descaled[-future:]
    })

    fig_predicted = go.Figure()
    fig_predicted.add_trace(go.Scatter(x=predicted_future['Date'], y=predicted_future['Open'], name='Predicted Close Price', line=dict(color='green')))
    fig_predicted.update_layout(title=f'{stock_name} - Predicted Future Prices', xaxis_title='Date', yaxis_title='Close Price')
    plot_div_predicted = fig_predicted.to_html(full_html=False)


    return render_template("index.html", stocks=stocks, plot_div_actual=plot_div_actual, plot_div_predicted=plot_div_predicted)

def train_models():
    for stock_name, df in stocks.items():
        lookback = 30
        future = 1
        X, y, _, _ = data_prep(df, lookback, future, Scale)
        model = Lstm_model1(X, y)
        models[stock_name] = model


if __name__ == "__main__":
    train_models()
    app.run(debug=True)
