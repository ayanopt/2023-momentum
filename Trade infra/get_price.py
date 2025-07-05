from logger import log
import json
import requests
from API.chalicelib.config import key
import yfinance as yf
def get_price(ticker):
    link = f"https://api.tdameritrade.com/v1/marketdata/{ticker}/quotes?apikey={key}%40AMER.OAUTHAP"
    try:
        body = requests.get(link)
        json_temp = json.loads(body.text)
        json_body = json_temp[ticker]
        price = json_body["askPrice"]
        return price
    except Exception as e:
        log(f"Error at {str(e)}",True)
        raise Exception(f"There has been an error in getting price for {ticker}")
def get_price_fx(ticker):
    try:
        ticker_yahoo = yf.Ticker(ticker)
        data = ticker_yahoo.history(period="max",interval="1m")
        last_quote = data['Close'].iloc[-1]
        return last_quote
    except Exception as e:
        log(f"Error at {str(e)}",True)
        raise Exception(f"There has been an error in getting price for {ticker}")