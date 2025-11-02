import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import threading
import time
import os
import pickle

class EnhancedParallelCryptoSimulatorCG:
    def __init__(self, starting_cash=1000.0, update_interval=10):
        self.starting_cash = starting_cash
        self.transaction_cost = 0.001
        self.update_interval = update_interval
        self.live_data = {}
        self.is_running = False
        self.data_lock = threading.Lock()
        self.output_dir = "portfolio_logs"
        os.makedirs(self.output_dir, exist_ok=True)

        self.params = {
            "MA_Original": {"window": 5, "max_position_pct": 0.15, "stop_loss_pct": 0.05, "take_profit_pct": 0.10},
            "MA_Fast": {"short_window": 3, "long_window": 10, "use_ema": True, "max_position_pct": 0.20, "stop_loss_pct": 0.08},
            "MA_Enhanced": {"short_window": 5, "long_window": 20, "volatility_threshold": 0.001, "max_position_pct": 0.12, "stop_loss_pct": 0.06, "take_profit_pct": 0.15},
            "Momentum_Enhanced": {"period": 1, "threshold": 0.5, "smoothing": 1, "max_position_pct": 0.9, "stop_loss_pct": 0.7},
            "Breakout": {"period": 5, "max_position_pct": 0.10, "stop_loss_pct": 0.04, "take_profit_pct": 0.12},
            "MACD": {"fast": 12, "slow": 26, "signal": 9, "max_position_pct": 0.15, "stop_loss_pct": 0.06},
            "ATR_Breakout": {"period": 14, "multiplier": 2.0, "max_position_pct": 0.12, "stop_loss_pct": 2.0, "take_profit_pct": 3.0},
            "ADX_Trend": {"period": 14, "min_strength": 25, "max_position_pct": 0.10, "stop_loss_pct": 0.08},
            "MeanReversion": {"period": 20, "buy_threshold": 0.98, "sell_threshold": 1.02, "max_position_pct": 0.08, "take_profit_pct": 1.015}
        }

    ##############################
    # Checkpoint methods
    ##############################
    def save_checkpoint(self, all_results, filename="checkpoint.pkl"):
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "wb") as f:
            pickle.dump(all_results, f)
        print(f"💾 Checkpoint saved to {filepath}")

    def load_checkpoint(self, filename="checkpoint.pkl"):
        filepath = os.path.join(self.output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            print(f"♻️ Checkpoint loaded from {filepath}")
            return data
        return {}

    ##############################
    # CoinGecko Data Fetching
    ##############################
    def start_real_time_data(self, coins):
        self.is_running = True
        self.coins = coins
        self.data_thread = threading.Thread(target=self._update_real_time_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        print(f"🔄 Started real-time CoinGecko data for {len(coins)} coins")

    def stop_real_time_data(self):
        self.is_running = False
        print("🛑 Stopped real-time data collection")

    def _update_real_time_data(self):
        while self.is_running:
            try:
                coin_ids = [c.split('-')[0].lower() for c in self.coins]
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(coin_ids)}&vs_currencies=usd"
                resp = requests.get(url).json()
                if 'error' in resp:
                    print(f"API Error: {resp['error']}")
                    time.sleep(60)
                    continue

                with self.data_lock:
                    for coin in self.coins:
                        cid = coin.split('-')[0].lower()
                        if cid in resp and 'usd' in resp[cid]:
                            price = float(resp[cid]['usd'])
                            if coin not in self.live_data:
                                self.live_data[coin] = {'history': pd.Series([price]), 'price': price, 'timestamp': datetime.now()}
                            else:
                                self.live_data[coin]['history'] = pd.concat([self.live_data[coin]['history'], pd.Series([price])]).tail(100)
                                self.live_data[coin]['price'] = price
                                self.live_data[coin]['timestamp'] = datetime.now()
                print(f"✅ Updated {len(self.coins)} coins at {datetime.now().strftime('%H:%M:%S')}")
            except Exception as e:
                print(f"❌ Error fetching data: {e}")
            time.sleep(self.update_interval)

    def get_current_prices(self):
        with self.data_lock:
            return {coin: data['price'] for coin, data in self.live_data.items() if 'price' in data}

    def get_price_history(self, coin, lookback=50):
        with self.data_lock:
            if coin in self.live_data:
                return self.live_data[coin]['history'].tail(lookback)
        return pd.Series(dtype=float)

    ##############################
    # Indicators & Signals
    ##############################
    def calculate_technical_indicators(self, prices, strategy):
        if len(prices) < 2: return {}
        indicators = {}
        if len(prices) > 10:
            returns = prices.pct_change()
            indicators['volatility'] = returns.std()
            indicators['momentum'] = (prices.iloc[-1]/prices.iloc[-10]-1) if len(prices)>=10 else 0
        if len(prices) > 14:
            delta = prices.diff()
            gain = (delta.where(delta>0,0)).rolling(14).mean()
            loss = (-delta.where(delta<0,0)).rolling(14).mean()
            rs = gain/loss
            indicators['rsi'] = 100 - (100/(1+rs)).iloc[-1]
        return indicators

    def generate_signal(self, prices, strategy):
        if prices.empty or len(prices)<2: return pd.Series([0]*len(prices))
        # Only implementing MA_Original here for brevity
        if strategy=="MA_Original":
            ma = prices.rolling(window=self.params[strategy]["window"]).mean()
            sig = (prices>ma).astype(int)
            return sig.replace({1:1,0:-1})
        return pd.Series([0]*len(prices))

    def enhanced_generate_signal(self, prices, strategy, indicators=None):
        if prices.empty or len(prices)<5: return 0
        base_signal = self.generate_signal(prices, strategy)
        signal_value = base_signal.iloc[-1] if not base_signal.empty else 0
        if indicators and 'rsi' in indicators:
            rsi = indicators['rsi']
            if signal_value>0 and rsi>70: return 0
            elif signal_value<0 and rsi<30: return 0
        return signal_value

    ##############################
    # Trading logic
    ##############################
    def calculate_dynamic_position_size(self, strategy, cash, current_volatility=None):
        base_pct = self.params[strategy].get("max_position_pct",0.1)
        if current_volatility and current_volatility>0.02: base_pct*=0.5
        return min(cash*base_pct, cash*0.95)

    def run_single_strategy(self, coins, strategy, duration_minutes=2):
        cash = self.starting_cash
        holdings = {coin:0 for coin in coins}
        entry_prices = {}
        portfolio_history = []
        trade_log = []
        end_time = datetime.now()+timedelta(minutes=duration_minutes)

        while datetime.now()<end_time and self.is_running:
            prices = self.get_current_prices()
            if len(prices)<len(coins)//2:
                time.sleep(1)
                continue
            signals = {}
            for coin in coins:
                if coin in prices:
                    ph = self.get_price_history(coin,100)
                    indicators = self.calculate_technical_indicators(ph,strategy)
                    signal = self.enhanced_generate_signal(ph,strategy,indicators)
                    signals[coin]=signal

            for coin,sig in signals.items():
                if sig>0 and cash>10:
                    size = self.calculate_dynamic_position_size(strategy,cash)
                    amount = size/prices[coin]
                    holdings[coin]+=amount
                    cash-=size
                    entry_prices[coin]=prices[coin]
                    trade_log.append({'timestamp':datetime.now(),'action':'BUY','coin':coin,'amount':amount,'price':prices[coin]})
            port_value = cash+sum(holdings.get(c,0)*prices.get(c,0) for c in coins)
            portfolio_history.append({'timestamp':datetime.now(),'value':port_value,'cash':cash,'holdings_value':port_value-cash})
            time.sleep(self.update_interval)

        final_value = portfolio_history[-1]['value'] if portfolio_history else self.starting_cash
        return {'Strategy':strategy,'Final Value':final_value,'Return':(final_value-self.starting_cash)/self.starting_cash*100,'Trades':len(trade_log),'Win Rate':0}, portfolio_history, trade_log

    ##############################
    # Parallel strategies
    ##############################
    def run_parallel_strategies(self, coins, strategies, duration_minutes=2):
        self.start_real_time_data(coins)
        time.sleep(5)
        all_results = self.load_checkpoint()
        threads=[]
        results_lock = threading.Lock()

        def worker(strategy):
            res, ph, log = self.run_single_strategy(coins,strategy,duration_minutes)
            with results_lock:
                all_results[strategy] = {'results':res,'portfolio_history':ph,'trade_log':log}
                self.save_checkpoint(all_results)

        for strategy in strategies:
            t = threading.Thread(target=worker,args=(strategy,))
            threads.append(t)
            t.start()
        for t in threads: t.join()
        self.stop_real_time_data()
        self.enhanced_plot_comparison(all_results)
        return all_results

    ##############################
    # Plotting
    ##############################
    def enhanced_plot_comparison(self, all_results):
        fig, ax = plt.subplots(figsize=(12,6))
        for strat,data in all_results.items():
            vals = [p['value'] for p in data['portfolio_history']]
            ax.plot(vals,label=strat)
        ax.set_title("Portfolio Value Comparison")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(True,alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'portfolio_comparison.png'))
        print("📈 Plot saved as portfolio_comparison.png")

    ##############################
    # Enhanced Export Methods
    ##############################
    def export_results(self, all_results, filename=None):
        if filename is None:
            filename = f"trading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_data=[]
        for strategy,data in all_results.items():
            results_data.append({'Strategy':strategy,'Final_Value':data['results']['Final Value'],'Return_Pct':data['results']['Return'],'Total_Trades':data['results']['Trades']})
        df = pd.DataFrame(results_data)
        df.to_csv(os.path.join(self.output_dir,filename),index=False)
        print(f"📊 Results exported to {filename}")
        return df

    def enhanced_export_all_data(self, all_results):
        """Export all simulation data including detailed trade logs"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export summary results
        summary_file = f"simulation_summary_{timestamp}.csv"
        self.export_results(all_results, summary_file)
        
        # Export detailed trade logs for each strategy
        for strategy, data in all_results.items():
            trades_file = f"trades_{strategy}_{timestamp}.csv"
            if data['trade_log']:
                trades_df = pd.DataFrame(data['trade_log'])
                trades_df.to_csv(os.path.join(self.output_dir, trades_file), index=False)
                print(f"📋 Trade log exported for {strategy}")
        
        # Export portfolio history for each strategy
        for strategy, data in all_results.items():
            portfolio_file = f"portfolio_{strategy}_{timestamp}.csv"
            if data['portfolio_history']:
                portfolio_df = pd.DataFrame(data['portfolio_history'])
                portfolio_df.to_csv(os.path.join(self.output_dir, portfolio_file), index=False)
                print(f"📈 Portfolio history exported for {strategy}")
        
        print(f"📊 All data exported with timestamp: {timestamp}")

# ------------------------
# Usage
# ------------------------
if __name__=="__main__":
    coins = [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD",
        "XRP-USD", "DOT-USD", "AVAX-USD", "LTC-USD",
        "ATOM-USD", "LINK-USD", "NEAR-USD", "HBAR-USD",
        "ICP-USD", "AR-USD", "AAVE-USD", "SAND-USD",
        "DOGE-USD", "SHIB-USD"
    ]
    strategies = [
        "ATR_Breakout",
        "Breakout",
        "Momentum_Enhanced",
        "MA_Original",
        "MA_Fast",
        "MA_Enhanced",
        "MACD",
        "ADX_Trend",
        "MeanReversion"
    ]

    simulator = EnhancedParallelCryptoSimulatorCG(starting_cash=10000, update_interval=15)
    print("🚀 Starting 5-Hour Enhanced Crypto Trading Simulation")
    all_results = simulator.run_parallel_strategies(coins, strategies, duration_minutes=300)  # 5 hours
    simulator.enhanced_export_all_data(all_results)  # Use enhanced export
    print("✅ 5-Hour Simulation completed! All logs saved!")