from tda import auth, client
import os, json, datetime
from chalice import Chalice
from chalicelib import config

app = Chalice(app_name='trading-view-options')
token_path = os.path.join(os.path.dirname(__file__), 'chalicelib', 'token.json')

c = auth.client_from_token_file(token_path, config.api_key)

@app.route('/specific/{specific}/{symbol}')
def specific(specific, symbol):
    response = c.get_quote(symbol)
    return (response.json())[symbol][specific]

@app.route('/quote/{symbol}')
def quote(symbol):
    response = c.get_quote(symbol)
    return response.json()

@app.route('/hello')
def index():
    return {'hello': 'world'} 

#@app.route('/get_{minute}_history/{symbol}')
#def history(symbol,minute):
#    exec_string = f"""
#    response = c.get_price_history_every_{minute}({symbol})
#    with open(f"../{symbol}_{minute}_data.txt","a") as f:
#        print(response.json(),file=f)
#    """
#    exec(exec_string)
#    
@app.route('/get_15m_history/{symbol}')
def history_15m(symbol):
    response = c.get_price_history_every_fifteen_minutes(symbol)
    with open(f"../{symbol}_15m_data.txt","a") as f:
        print(response.json(),file=f)
@app.route('/get_1m_history/{symbol}')
def history_1m(symbol):
    response = c.get_price_history_every_minute(symbol)
    with open(f"../{symbol}_1m_data.txt","a") as f:
        print(response.json(),file=f)



"""@app.route('/option/order', methods=['POST'])
def option_order():

    webhook_message = app.current_request.json_body

    print(webhook_message)


    if 'passphrase' not in webhook_message:
        return {
            "code": "error",
            "message": "Unauthorized, no passphrase"
        }

    if webhook_message['passphrase'] != config.passphrase:
        return {
            "code": "error",
            "message": "Invalid passphrase"
        }

    order_spec = {
        "orderType": "MARKET",
        "session": "NORMAL",
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [
            {
                "instruction": webhook_message["instruction"],
                "quantity": webhook_message["quantity"],
                "instrument": {
                    "symbol": webhook_message["symbol"],
                    "assetType": "OPTION"
                }
            }
        ]
    }




    response = c.place_order(config.account_id, order_spec)
    return {
        "code": "ok"
    }

# The view function above will return {"hello": "world"}
# whenever you make an HTTP GET request to '/'.
#
# Here are a few more examples:
#
# @app.route('/hello/{name}')
# def hello_name(name):
#    # '/hello/james' -> {"hello": "james"}
#    return {'hello': name}
#
# @app.route('/users', methods=['POST'])
# def create_user():
#     # This is the JSON body the user sent in their POST request.
#     user_as_json = app.current_request.json_body
#     # We'll echo the json body back to the user in a 'user' key.
#     return {'user': user_as_json}
#
# See the README documentation for more examples.
#
"""