from alpha_vantage.fundamentaldata import FundamentalData
from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import yfinance as yf
import stocknews as sn
import matplotlib.pyplot as plt
from streamlit_option_menu import *
import plotly_express as px
import plotly.graph_objects as go
from function import *
import requests
from stocknews import StockNews
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

ticker = 'BBRI.JK'
st.title('Dashboard Saham BBRI')
def fetch_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

def plot_candlestick(data):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])

    fig.update_layout(title='Candlestick Chart for BBRI',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)

    return fig

start_date = st.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('today'))

if start_date < end_date:
      # BBRI stock ticker on Yahoo Finance
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    if not stock_data.empty:
        st.plotly_chart(plot_candlestick(stock_data), use_container_width=True)
    else:
        st.warning('No data available for the selected date range.')
else:
    st.error('Error: End date must be after start date.')

st.sidebar.image('asset/LogoBRI.png', width=200)

tab1, tab2, tab3, tab4 = st.tabs(["Pricing Data", "ALL About BBRI", "Top 10 News", "Predict Price"])

with tab1:
    st.header('Price Movements')
    stock_data2 = stock_data
    stock_data2['% Change'] = stock_data['Adj Close'] / stock_data['Adj Close'].shift(1)-1
    stock_data2.dropna(inplace = True)
    st.write(stock_data2)
    annual_return=stock_data2['% Change'].mean()*252*100
    st.write('Annual Return is', annual_return,'%')
    stdev = np.std(stock_data2['% Change'])*np.sqrt(252)
    st.write('Standard Deviation is', stdev*100,'%')
    st.write('Risk Adj. Return is', annual_return/(stdev*100))

with tab2:
    bbri= yf.Ticker('BBRI.JK')
    st.write(bbri.info)

with tab3:
    st.header(f'News of {ticker}')
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader (f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')

with tab4:

    # Function to load and preprocess the data
    def load_data(start_date, end_date):
        # Load data from Yahoo Finance
        df = yf.download("BBRI.JK", start=start_date, end=end_date)
        df.reset_index(inplace=True)
        return df

    # Function to create features
    def create_features(df):
        # Create lag feature
        df['Close_Lag1'] = df['Close'].shift(1)
        df.dropna(inplace=True)

        # Add column indicating whether the price went up or down
        df['Naik_Turun'] = (df['Close'] - df['Close_Lag1']).apply(lambda x: 1 if x > 0 else 0)

        return df

    # Function to train a Random Forest model
    def train_model(df):
        X = df[['Close_Lag1']]
        y = df['Naik_Turun']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy: {accuracy}')

        return model

    # Function to make predictions using the trained model
    def make_predictions(model, df, future_days=30):
        last_date = df.iloc[-1]['Date']
        future_dates = pd.date_range(start=last_date, periods=future_days+1)[1:]

        future_X = df['Close'].tail(1).values  # Use the last close price as input
        future_preds = []
        for _ in range(future_days):
            future_pred = model.predict([future_X[-1:]])[0]
            future_preds.append(future_pred)
            future_X = np.roll(future_X, -1)
            future_X[-1] = future_pred

        future_df = pd.DataFrame({'Date': future_dates, 'Predicted Naik_Turun': future_preds})
        return future_df

    # Main function to run the Streamlit app
    def main():
        st.title('Prediksi Saham BBRI.JK menggunakan Streamlit')

        # Get today's date and 30 days from today
        today = datetime.today().date()
        future_date = today + timedelta(days=30)

        # Load data
        data_load_state = st.text('Loading data...')
        df = load_data(today - timedelta(days=365), future_date)
        data_load_state.text('Data loaded successfully!')

        # Create features
        df = create_features(df)

        # Train model
        model_train_state = st.text('Training model...')
        model = train_model(df)
        model_train_state.text('Model trained successfully!')

        # Make predictions
        future_df = make_predictions(model, df)

        # Display data
        st.subheader('Data Saham BBRI.JK')
        st.write(df)

        # Display predictions
        st.subheader('Prediksi Naik_Turun Harga Saham BBRI.JK untuk 30 Hari ke Depan')
        st.write(future_df)

    if __name__ == '__main__':
        main()

# saham=('BBRI')

# tab1, tab2, tab3 = st.tabs(["View", "Pricing Data", "Fundamental Data"])
# with st.sidebar :
#     selected = option_menu('BBRI STOCK',['Introducing','Data Distribution','Relation','Composition & Comparison','Predict','Clustering'],default_index=0)

# with tab1:
#    sd = st.sidebar.date_input('Start Date')
#    ed = st.sidebar.date_input('End Date')
#    data = yf.download(saham,start=sd,end=ed)
#     # Buat candlestick chart
#     fig = go.Figure(data=[go.Candlestick(x=df['Date'], # type: ignore
#                     open=df['Open'],
#                     high=df['High'],
#                     low=df['Low'],
#                     close=df['Close'])])

#     # Konfigurasi layout
#     fig.update_layout(title='Candlestick Chart',
#                       xaxis_title='Date',
#                       yaxis_title='Stock Price (IDR)',
#                       xaxis_rangeslider_visible=False,
#                       template='plotly_dark')



# df = pd.read_csv('https://raw.githubusercontent.com/eyotaa/MiniProject_DataMining/main/checkpoin%201/BBRI.JK.csv')
# dff = pd.read_csv("https://raw.githubusercontent.com/eyotaa/MiniProject_DataMining/main/checkpoin3/data%20clean.csv")
# dfff = pd.read_csv("https://raw.githubusercontent.com/eyotaa/MiniProject_DataMining/main/checkpoin5/data%20final.csv")

# with st.sidebar :
#     selected = option_menu('BBRI STOCK',['Introducing','Data Distribution','Relation','Composition & Comparison','Predict','Clustering'],default_index=0)

# # Introduction Page


# if (selected == 'Introducing'):
#             # Buat candlestick chart
#     fig = go.Figure(data=[go.Candlestick(x=df['Date'], # type: ignore
#                     open=df['Open'],
#                     high=df['High'],
#                     low=df['Low'],
#                     close=df['Close'])])

#     # Konfigurasi layout
#     fig.update_layout(title='Candlestick Chart',
#                       xaxis_title='Date',
#                       yaxis_title='Stock Price (IDR)',
#                       xaxis_rangeslider_visible=False,
#                       template='plotly_dark')

#     # Tampilkan judul
#     st.title('Introduction')

#     # Tampilkan pengantar
#     st.write("""
#     Di sini, kami memperlihatkan sebuah visualisasi Candlestick Chart yang menggambarkan pergerakan harga saham. 
#     Candlestick chart merupakan jenis chart yang banyak digunakan dalam analisis pasar keuangan untuk melihat fluktuasi harga saham dari waktu ke waktu.
#     """)

#     # Tampilkan visualisasi
#     st.plotly_chart(fig)
    
# if (selected == 'Data Distribution'):
#     st.header("Data Distribution")
#     scatter_plot(df)
    
# if (selected == 'Relation'):
#     st.title('Relations')
#     heatmap(df)

# if (selected == 'Composition & Comparison'):
#     st.title('Composition')
#     compositionAndComparison(df)

# if (selected == 'Predict'):
#     st.title('Let\'s Predict!')
#     predict()

# if (selected == 'Clustering'):
#     st.title('Clustering!')
#     clustering(df)

