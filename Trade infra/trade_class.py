class trade:
    def __init__(self, profit_in, loss_in, bot_price_in, qty_in,strat_in,timeout_in) -> None:
        self.profit_price = profit_in
        self.loss_price = loss_in
        self.bot_price = bot_price_in
        self.timeout = timeout_in
        self.qty = qty_in
        self.strat = strat_in
    def tick(self,price):
        if price >= self.profit_price or price <= self.loss_price or self.timeout == 0:
            return price-self.bot_price
        self.timeout-=1
        return None