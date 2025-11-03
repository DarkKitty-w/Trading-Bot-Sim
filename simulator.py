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
            "MA_Original": {"window": 5, "max_position_pct": 0.15, "stop_loss_pct": 0.05, "take_profit_pct": 0.10, "min_price_variation_pct": 0.0003, "confirmation_periods": 1, "timeframes": ['1m', '5m']}, # Lowered min_price_variation_pct
            "MA_Fast": {"short_window": 3, "long_window": 10, "use_ema": True, "use_kama": False, "kama_fast_period": 2, "kama_slow_period": 30, "max_position_pct": 0.20, "stop_loss_pct": 0.08, "min_price_variation_pct": 0.0003, "confirmation_periods": 1, "timeframes": ['1m', '5m', '15m']}, # Lowered min_price_variation_pct, reduced confirmation_periods
            "MA_Enhanced": {"short_window": 5, "long_window": 20, "volatility_threshold": 0.0005, "max_position_pct": 0.12, "stop_loss_pct": 0.06, "take_profit_pct": 0.15, "min_price_variation_pct": 0.0005, "confirmation_periods": 1, "timeframes": ['1m', '5m']}, # Lowered volatility_threshold and min_price_variation_pct, reduced confirmation_periods
            "Momentum_Enhanced": {"period": 1, "threshold": 0.3, "smoothing": 1, "max_position_pct": 0.9, "stop_loss_pct": 0.7, "min_price_variation_pct": 0.0005, "confirmation_periods": 1, "timeframes": ['1m']}, # Lowered threshold
            "Breakout": {"period": 5, "max_position_pct": 0.10, "stop_loss_pct": 0.04, "take_profit_pct": 0.12, "min_price_variation_pct": 0.0005, "confirmation_periods": 1, "timeframes": ['1m', '15m']}, # Lowered min_price_variation_pct, reduced confirmation_periods
            "MACD": {"fast": 12, "slow": 26, "signal": 9, "max_position_pct": 0.15, "stop_loss_pct": 0.06, "min_price_variation_pct": 0.0005, "confirmation_periods": 1, "timeframes": ['1m', '5m']}, # Lowered min_price_variation_pct, reduced confirmation_periods
            "ATR_Breakout": {"period": 14, "multiplier": 1.5, "max_position_pct": 0.12, "stop_loss_pct": 2.0, "take_profit_pct": 3.0, "min_price_variation_pct": 0.0008, "confirmation_periods": 1, "timeframes": ['1m', '15m']}, # Lowered multiplier and min_price_variation_pct, reduced confirmation_periods
            "ADX_Trend": {"period": 14, "min_strength": 20, "max_position_pct": 0.10, "stop_loss_pct": 0.08, "min_price_variation_pct": 0.0005, "confirmation_periods": 1, "timeframes": ['1m', '5m']}, # Lowered min_strength and min_price_variation_pct, reduced confirmation_periods
            "MeanReversion": {"period": 20, "buy_threshold": 0.99, "sell_threshold": 1.01, "max_position_pct": 0.08, "take_profit_pct": 1.015, "min_price_variation_pct": 0.0003, "confirmation_periods": 1, "limit_buy_offset_pct": -0.003, "trailing_stop_pct": 0.02, "timeframes": ['1m', '5m']} # Adjusted thresholds, min_price_variation_pct, limit_buy_offset_pct, and trailing_stop_pct, reduced confirmation_periods
        }
        self.coingecko_id_map = {
         "BTC-USD": "90",
         "ETH-USD": "80",
         "XRP-USD": "58",
         "BNB-USD": "2710",
         "SOL-USD": "48543",
         "DOGE-USD": "2",
         "ADA-USD": "257",
         "LINK-USD": "2751",
         "HBAR-USD": "48555",
         "AVAX-USD": "44883",
         "LTC-USD": "1",
         "SHIB-USD": "45088",
         "DOT-USD": "45219",
         "AAVE-USD": "46018",
         "NEAR-USD": "48563",
         "ICP-USD": "47311",
         "ATOM-USD": "33830",
         "SAND-USD": "45161",
         "AR-USD": "42441"
     }
        # Dictionaries to store historical data at different granularities
        self.historical_data = {
            '1m': {},
            '5m': {},
            '15m': {}
        }
        self._last_update_time = None

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
    # CoinGecko Data Fetching - REPAIRED VERSION
    ##############################
    def start_real_time_data(self, coins):
        self.is_running = True
        self.coins = coins
        self.data_thread = threading.Thread(target=self._update_real_time_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        print(f"🔄 Started real-time CoinLore data for {len(coins)} coins")

    def stop_real_time_data(self):
        self.is_running = False
        print("🛑 Stopped real-time data collection")

    def _update_real_time_data(self):
        while self.is_running:
            try:
                # Get valid CoinLore IDs (on réutilise coingecko_id_map pour les IDs CoinLore)
                coin_ids_for_api = []
                valid_coins = []

                for coin in self.coins:
                    if coin in self.coingecko_id_map:
                        coin_ids_for_api.append(self.coingecko_id_map[coin])
                        valid_coins.append(coin)
                    else:
                        print(f"⚠️ Warning: {coin} not found in CoinLore mapping (in coingecko_id_map)")

                if not coin_ids_for_api:
                    print("❌ Error: No valid coins to fetch.")
                    time.sleep(60)
                    continue

                # Construire URL pour CoinLore : /api/ticker/?id=ID1,ID2,...
                ids_param = ",".join(coin_ids_for_api)
                url = f"https://api.coinlore.net/api/ticker/?id={ids_param}"

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }

                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    print(f"❌ API Error (CoinLore): Status code {response.status_code}")
                    print("DEBUG: response.text:", response.text)
                    time.sleep(60)
                    continue

                resp = response.json()
                # resp est une liste d’objets JSON, ex : [ { "id":"90", "symbol":"BTC", "price_usd":"12345.67", … }, … ]

                current_time = datetime.now()

                with self.data_lock:
                    updated_count = 0
                    for coin in valid_coins:
                        lore_id = self.coingecko_id_map[coin]
                        obj = None
                        for item in resp:
                            if str(item.get("id")) == str(lore_id):
                                obj = item
                                break

                        if obj is None:
                            print(f"⚠️ No JSON object for {coin} with id {lore_id}")
                            continue

                        price_usd = obj.get("price_usd")
                        if price_usd is None:
                            print(f"⚠️ price_usd missing for {coin}: {price_usd}")
                            continue

                        try:
                            price = float(price_usd)
                        except ValueError:
                            print(f"⚠️ Cannot convert price_usd to float for {coin}: {price_usd}")
                            continue

                        # print(f"🪙 {coin}: ${price:,.2f} USD (CoinLore)") # Suppress frequent logging

                        if coin not in self.live_data:
                            self.live_data[coin] = {
                                'history': pd.Series([price], index=[current_time]),
                                'price': price,
                                'timestamp': current_time
                            }
                            # Initialize historical_data for the coin
                            for timeframe in self.historical_data.keys():
                                self.historical_data[timeframe][coin] = pd.Series([price], index=[current_time])
                        else:
                            new_data = pd.Series([price], index=[current_time])
                            self.live_data[coin]['history'] = pd.concat([
                                self.live_data[coin]['history'],
                                new_data
                            ])#.tail(100) # Keep all data for downsampling
                            self.live_data[coin]['price'] = price
                            self.live_data[coin]['timestamp'] = current_time

                            # Update historical_data at different granularities
                            for timeframe, data_series_dict in self.historical_data.items():
                                # Determine the interval in minutes
                                interval_minutes = int(timeframe[:-1])
                                # Check if the current time aligns with the timeframe interval
                                if self._last_update_time is None or (current_time - self._last_update_time).total_seconds() >= interval_minutes * 60:
                                     # Append the current price if enough time has passed or it's the first update
                                     data_series_dict[coin] = pd.concat([
                                         data_series_dict.get(coin, pd.Series([], dtype=float)), # Handle initial case
                                         pd.Series([price], index=[current_time])
                                     ])


                        updated_count += 1

                    if updated_count > 0:
                         self._last_update_time = current_time # Update last update time only if data was successfully updated
                         print(f"✅ Updated {updated_count}/{len(valid_coins)} coins at {current_time.strftime('%H:%M:%S')}")


            except requests.exceptions.RequestException as e:
                print(f"❌ Network error fetching data (CoinLore): {e}")
                time.sleep(30)
            except Exception as e:
                print(f"❌ Unexpected error in data update (CoinLore): {e}")
                import traceback
                traceback.print_exc()
                time.sleep(30)

            time.sleep(self.update_interval)

    def get_current_prices(self):
        """Récupère les prix les plus récents de live_data."""
        with self.data_lock:
            # retourne un dict {coin: price}
            return {coin: data['price'] for coin, data in self.live_data.items() if 'price' in data}

    def get_price_history(self, coin, timeframe='1m', limit=None):
        """
        Récupère l’historique des prix pour un coin à une granularité donnée, limité si besoin.
        timeframe: '1m', '5m', '15m'
        """
        with self.data_lock:
            if timeframe in self.historical_data and coin in self.historical_data[timeframe]:
                history = self.historical_data[timeframe][coin]
                if limit is not None:
                    return history.tail(limit)
                return history
            # Fallback to the most granular data if requested timeframe is not available
            elif coin in self.live_data and 'history' in self.live_data[coin]:
                 print(f"⚠️ Requested timeframe '{timeframe}' not available for {coin}, using full history.")
                 history = self.live_data[coin]['history']
                 if limit is not None:
                     return history.tail(limit)
                 return history
            return pd.Series([]) # retourne une série vide si pas de données

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
