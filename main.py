import streamlit as st 
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
from yahooquery import search
import yfinance as yf
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
import locale
from arima import predict_stock_action
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# LODING API KEY
load_dotenv()
locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')

# GEMINI AI MODEL OBJECT
apiKey = os.getenv("API_KEY")
api_key = apiKey
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")


# GEMINI API CALL - 1 : FOR TEXT GENERATION 
def ai_call(stocks, actual_price, predicted_price, sentment_score=0): 
        prompt = f'Write a short paragraph advising whether or not to buy {stocks} stock based on the following information: * current market price = {actual_price} * Predicted trend for TCS is = {predicted_price}. The advice should be concise and clear, focusing on the key points with current market sentiment'
        response = model.generate_content(prompt)
        return response.text


# BUY AND SELL INDICATIOR CSS
def indicator(label, color):
    custom_html = f"""
    <div style="text-align: center;">
        <button style="
            background-color: {color};
            width: 100%;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;">
            {label}
        </button>
    </div>
    """
    st.markdown(custom_html, unsafe_allow_html=True)


# STOCK SYMBOL RETRIVE FUNCTION
def get_stock_symbol(company_name):
    symbol = ''
    response = search(company_name)
    for result in response['quotes']:
        if '.NS' in result['symbol']:
            symbol = result['symbol']
            break
        print(symbol)

    return symbol



# SENTIMENT ANALYZER OBJECT
analyzer = SentimentIntensityAnalyzer()

# SENTIMENT ANALYZER FUNCTION
def get_sentiment_review(text):
    sentiment = analyzer.polarity_scores(text)
    compound_score = sentiment['compound']

    if compound_score >= 0.05:
        review = "Positive"
    elif compound_score <= -0.05:
        review = "Negative"
    else:
        review = "Neutral"
    
    normalized_score = int((compound_score + 1) * 50)
    return normalized_score, review



# GRAPH PLOTING 
def plot_graph(ticker_symbol):
    
    if ticker_symbol:
        try:
            # Fetch stock data
            stock_data = yf.Ticker(ticker_symbol)

            # Calculating previous day date
            today_date = datetime.now()
            previous_day = (today_date - timedelta(days=1)).strftime('2024-12-13')

            # Fetch data for the previous last
            historical_data = stock_data.history(period="1d", interval="1m")

            # Reset index to access datetime
            historical_data = historical_data.reset_index()
            historical_data['Datetime'] = pd.to_datetime(historical_data['Datetime'])

            # Filter for the previous day's data
            historical_data = historical_data[
                historical_data['Datetime'].dt.date == pd.to_datetime(previous_day).date()
            ]

            

            if not historical_data.empty:
                # Select Datetime and Close columns
                plot_data = historical_data[['Datetime', 'Close']]

                fig = go.Figure(data=[go.Candlestick(
                    x=historical_data['Datetime'],
                    open=historical_data['Open'],
                    high=historical_data['High'],
                    low=historical_data['Low'],
                    close=historical_data['Close'],
                    increasing_line_color='green',
                    decreasing_line_color='red'
                )])

                fig.update_layout(
                    title=f'{ticker_symbol.upper()}',
                    xaxis_title='Time',
                    yaxis_title='Stock Price (INR)',
                    xaxis_rangeslider_visible=False
                )

                # CALLING STACK PRIDICTION MODULE 
                last_price, predicted_price, action = predict_stock_action(ticker_symbol)

                # price = historical_data['Open']
                
                current_price = float(last_price)
                formatted_amount = locale.currency(current_price, symbol=True, grouping=True)

                # GRAPH PLOTED FROM HERE 
                st.write(' ') 
                st.write(' ')
                st.header(formatted_amount)

                # Display graph in Streamlit
                st.plotly_chart(fig, use_container_width=True)

                
                # BUY AND SELL INDICATOR
                incol1, incol2 = st.columns(2)
                st.write(' ') 
                st.write(' ')
                st.subheader('Buy and Sell indicator')

                with incol1:
                    st.write(f'Actual price    : {locale.currency(last_price, symbol=True, grouping=True)}')
                
                with incol2: 
                    st.write(f'Predicted price : {locale.currency(predicted_price, symbol=True, grouping=True)}')

                
                # BUY OR SELL INDICATOR
                ans = ai_call(ticker_symbol, formatted_amount, predicted_price)
                sentiment_scores, review = get_sentiment_review(ans)

                if action.lower() == 'buy': 
                    indicator(f'{action.upper()}', '#1dcf46')
                     
                else:
                    indicator(f'{action.upper()}', 'red')


                # DISPLAYING RAW DATA OF 5
                st.write(' ') 
                st.write(' ')
                # Display raw data
                st.subheader('Historic Data')
                st.dataframe(historical_data.head())


                # SENTIMENT DATA PRINTING
                st.write(' ') 
                st.write(' ')
                st.subheader('Sentiment Data')
                sed1 , sed2 = st.columns(2)
                
                with sed1:
                    st.markdown(f'Score : {sentiment_scores}')

                with sed2:
                    st.markdown(f'Indication : {review}')

                # AI SUGGESTION ON STOCK
                ai_answer = ai_call(ticker_symbol, formatted_amount, predicted_price, sentiment_scores)
                st.write(' ') 
                st.write(' ')
                st.subheader('AI Suggestion ✨')
                st.write(ai_answer)

            else:
                st.warning("No data available for the previous day.")

        except Exception as e:
            st.error(f"An error occurred: {e}")







st.title('InvestIQ')
st.text('Smart Insights, Smarter Investments')
st.write(' ')
st.write(' ')


# WARNING OR ERROR MESSAGE 
msg = st.empty()

# STOCK EXCHANGE LIST 
stock_exchage = ['NSE' , 'BSE'] 

# SEARCH BAR COLUMN 
scol1, scol2 = st.columns(2)

# SEARCH BUTTON COLUMN
btncol1, btncol2 = st.columns(2)

# STOCK INPUT 
with scol1:
    userInput = st.text_input('Enter stock name ')

# SCTOCK EXCHANGE
with scol2:
    pass
    stock = stock_option = st.selectbox("Select stock exchage", stock_exchage)


ans = ''
with btncol1:
     if st.button('Submit', use_container_width=True): 
        if userInput == "":
            msg.warning('⚠️ Please entre stock name !')
        else:
            ans = get_stock_symbol(userInput)
            if ans == '':
                ans = ans.replace('.BO', '.NS')
                msg.warning('⚠️ Enter valid company name !')
            else:
                pass

               
                

st.write('')                
plot_graph(ans)


st.write('') 
st.write('') 
st.write('\n\n') 
st.divider()
st.subheader('TEAM-BCU')
st.write('1. Praveen Kumar\n2. Santosh\n3. Skanda Umesh\n4. Ankith Kumar')