#---------------------------------------------local trading deps
import statistics as st
import pandas as pd
from trade_class import trade
from logger import log,clear_log
import os, math, time, subprocess, datetime
from get_price import get_price

#---------------------------------------------initial capital and drawdown
trades = {}
SMAs = [7,20,50,100]
capital = {"1_3":[], "3_5":[], "5_15":[]}
trade_counter = 0
for key in capital:
    f = open(f"./data/capital/capital_{key}.txt","r")
    capital[key].append(float(f.read()))
    f.close()
    
#----------------------------------------------fail safe
n_fails = 0

#----------------------------------------------delete log file content
clear_log()
#--------------------------------------------start initialization
from load_initial import past_prices, ticker, minute_interval
    
#---------------------------------------------Account for the initialization
for it in range((8)*60):
    naur = datetime.datetime.now()
    if naur.hour == 16 and naur.minute >= 13:
        break
    
    #-----------------------------------------load price
    try:
        price = get_price(ticker)*10
        
    except Exception as e:
        n_fails+=1
        log(str(e),True)
        if n_fails == 5:
            break
        time.sleep(60)
        continue
    
    #-----------------------------------------find trades to close
    remove_trade_indices = []
    for trade_id in trades:
        curr_trade = trades[trade_id]
        curr_strat = curr_trade.strat
        result = curr_trade.tick(price)
        if result is None:
            continue
        remove_trade_indices.append(trade_id)
        PL = curr_trade.qty*result - (1.3)
        
        #-------------------------------------log the trade
        log(f"***SOLD {curr_trade.qty} {ticker} at {price} trade id: {trade_id} with profit/loss {PL}", True)
        log(f"--strat {curr_trade.strat}",False)
        log(f"Was it a timeout close? {curr_trade.timeout == 0}",False)
        capital[curr_strat]+=[capital[curr_strat][-1]+PL]
        
    #-----------------------------------------pop all closed trades
    while len(remove_trade_indices)>=1:
        trades.pop(remove_trade_indices[-1])
        remove_trade_indices.pop()

    #-----------------------------------------indicators
    curr_atr = st.stdev(past_prices[-15:])
    
    df = pd.DataFrame()
                
        
    for k1 in range(len(SMAs)):
        curr_SMA = SMAs[k1]
        curr_SMA_val =  price/st.mean(past_prices[-curr_SMA: ])
        df.loc[0,f"SMA_k{curr_SMA}"] = curr_SMA_val
        colname1 = f"prev1.{curr_SMA}"
        colname2 = f"prev2.{curr_SMA}"
        
        prev1 = ((past_prices[-1]/st.mean(past_prices[-curr_SMA-1: -1]))-curr_SMA_val)*10_000
        prev2 = ((past_prices[-2]/st.mean(past_prices[-curr_SMA-2: -2]))-curr_SMA_val)*10_000
        if 0 in [prev1,prev2]:
            prev1 = -1000
            prev2 = -1000
        df.loc[0,colname1] = prev1
        df.loc[0,colname2] = prev2
        df.loc[0,f"{colname1}.{colname2}"] = prev1/prev2
        
        for k2 in range(k1-1,-1,-1):
            prev_SMA = SMAs[k2] 
            prev2x_1 = f"prev1.{prev_SMA}"
            prev2x_2 = f"prev2.{prev_SMA}"
            
            df.loc[0,f"{prev2x_1}.{colname1}"] = df.loc[0,prev2x_1]/prev1
            df.loc[0,f"{prev2x_1}.{colname2}"] = df.loc[0,prev2x_1]/prev2
            df.loc[0,f"{prev2x_2}.{colname1}"] = df.loc[0,prev2x_2]/prev1
            df.loc[0,f"{prev2x_2}.{colname2}"] = df.loc[0,prev2x_2]/prev2
            
            df.loc[0,f"r{prev_SMA}.{curr_SMA}"] =  (df.loc[0,f"SMA_k{prev_SMA}"]/curr_SMA_val) * 100
            
    
    if len(past_prices) > 102:
        past_prices.pop(0)
    past_prices.append(price)

    df.loc[0,"ATR"] = curr_atr
    
    df.to_csv(f"./{ticker}_{minute_interval}m_current.csv")
    
    #-----------------------------------------execute model
    subprocess.call (["/usr/bin/Rscript", "--vanilla", 
                        f"./{ticker}_{minute_interval}m_load_model.r"])
    
    #-----------------------------------------trade models
    for key in capital:
        ot = pd.read_csv(f"./{ticker}_{minute_interval}m_{key}_out.csv")
        exec_trade = True
        
        qty = 10
        
        parser = key.split("_")
        chi = int(parser[0])
        timeout = int(parser[1])

        profit_price = price + 1.5 * chi * curr_atr
        loss_price = price - chi * curr_atr
        
        #--------------------------------------model specifics check
        for col in ot.columns:
            exec_trade = exec_trade and ot.loc[0,col] > 0
        
        #--------------------------------------buy current candle
        if exec_trade:
            new_trade = trade(profit_price,loss_price,price,qty,key,timeout)
            trade_counter += 1
            trades[trade_counter] = new_trade
            log(f"""***BOT {new_trade.qty} {ticker} at {price}, TP: {profit_price},
                                        SL: {loss_price}, id: {trade_counter} 
                                            --strat {new_trade.strat}""",True)
    out_str = f"iteration {it} of trading, current capital: "
    for key in capital:
        out_str += "\n\t"
        out_str += f"--strat {str(key)}: {capital[key][-1]}"
        
        #-------------------------------------remove files and repeat in 5 mins
        #os.remove(f"./{ticker}_{minute_interval}m_{key}_out.csv")
    log(out_str, True)
        
    #os.remove(f"./{ticker}_{minute_interval}m_current.csv")
    time.sleep(51.5)
    
#---------------------------------------------pop all trades EOD
for trade_id in trades:
    curr_trade = trades[trade_id]
    curr_strat = curr_trade.strat
    curr_trade.timeout = 0
    result = curr_trade.tick(price)
    PL = curr_trade.qty*result - 1.3
    
    #----------------------------------------log the trade
    log(f"***SOLD {curr_trade.qty} {ticker} trade id: {trade_id} with profit/loss {PL}", True)
    log(f"--strat {curr_trade.strat}",False)
    log("EOD CLOSE",False)
    capital[curr_strat]+=[capital[curr_strat][-1]+PL]
    
#---------------------------------------------Log the capital
for key in capital:
    f = open(f"./data/capital/capital_{key}.txt", "w")
    f.write(str(capital[key][-1]))
    f.close()   
    log(f"End: {capital[key][-1]}",True)
    
subprocess.call(["git","add","."])
date_time_rn = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
subprocess.call(["git","commit","-m",f'"{date_time_rn}"'])
subprocess.call(["git","push"])
