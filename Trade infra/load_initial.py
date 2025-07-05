# hi from ubuntu~
import time
from logger import log
from get_price import get_price
past_prices = []

#---------------------------------------------load all indicators
json_body = None
ticker = "SPY"
minute_interval = 1
n_fails = 0
for i in range(103):
    try:
        price = get_price(ticker)*10
        
    except Exception as e:
        n_fails+=1
        log(str(e),True)
        if n_fails == 5:
            break
        time.sleep(60)
        continue
    if len(past_prices)>102:
        past_prices.pop(0)
    past_prices.append(price)

    #-----------------------------------------log tick and repeat
    log(f"iteration {i} of initialization {ticker} at {minute_interval}m", True)
    time.sleep((60 * minute_interval) - 0.2)
