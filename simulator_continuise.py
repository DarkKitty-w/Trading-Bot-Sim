import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import requests
import os
from datetime import datetime, timedelta
import concurrent.futures
import scipy.optimize as sco
import logging
from typing import Dict, List, Optional, Tuple
import json
import hashlib
import warnings
from dataclasses import dataclass
from enum import Enum
import gc
from logging.handlers import RotatingFileHandler
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib
import copy  # <-- ADDED IMPORT

matplotlib.use('Agg')  # Pour meilleures performances

class AdvancedChartGenerator:
    """Générateur de graphiques avancés with visualisations interactives"""
    
    def __init__(self, output_dir="portfolio_logs_10"):
        self.output_dir = output_dir
        os.makedirs(os.path.join(self.output_dir, "charts"), exist_ok=True)
        self.setup_matplotlib_styles()
        
    def setup_matplotlib_styles(self):
        """Configure les styles matplotlib pour des graphiques professionnels"""
        plt.style.use('seaborn-v0_8')
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#18A558',
            'danger': '#F24236',
            'warning': '#F5B700',
            'info': '#4FB0C6',
            'dark': '#2F2F2F',
            'light': '#F8F9FA'
        }
        
        # Palette de couleurs pour multiples séries
        self.palette = [
            '#2E86AB', '#A23B72', '#18A558', '#F24236', '#F5B700',
            '#4FB0C6', '#6A4C93', '#FF6B6B', '#4ECDC4', '#45B7D1'
        ]

    # ... (All other methods of AdvancedChartGenerator remain unchanged) ...
    # ... (plot_cumulative_returns, plot_trade_analysis, etc.) ...
    
    def plot_trade_analysis(self, all_results: Dict):
        """Analyse des trades"""
        # ... (Method content unchanged) ...
        
class DataFetchError(Exception):
    """Exception personnalisée pour les erreurs de fetch de données"""
    pass

class OrderType(Enum):
    MARKET_BUY = 1
    MARKET_SELL = 2
    LIMIT_BUY = 3
    LIMIT_SELL = 4
    STOP_LOSS = 5
    TAKE_PROFIT = 6

@dataclass
class Order:
    id: str
    coin: str
    type: OrderType
    amount: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    oco_group: Optional[str] = None
    created_at: datetime = datetime.now()

class DataFetcher:
    """Gère la récupération des données de marché"""
    
    def __init__(self, coinlore_id_map: Dict, update_interval=5, max_history_points=1440): # 1440 points * 1 min = 24h
        self.coinlore_id_map = coinlore_id_map
        self.coinlore_reverse_map = {v: k for k, v in coinlore_id_map.items()}
        self.update_interval = update_interval
        self.max_history_points = max_history_points
        self.data_lock = threading.Lock()
        self.is_running = False
        
        self.live_data: Dict[str, float] = {}
        self._last_update_time: Optional[datetime] = None
        
        # Historique des prix pour les indicateurs (1m, 5m, 15m, 1h)
        self.historical_data_dfs: Dict[str, pd.DataFrame] = {
            '1m': pd.DataFrame(columns=list(coinlore_id_map.keys())),
            '5m': pd.DataFrame(columns=list(coinlore_id_map.keys())),
            '15m': pd.DataFrame(columns=list(coinlore_id_map.keys())),
            '1h': pd.DataFrame(columns=list(coinlore_id_map.keys()))
        }
        
        self._last_timeframe_update: Dict[str, Optional[datetime]] = {
            '1m': None, '5m': None, '15m': None, '1h': None
        }

    # ... (All methods of DataFetcher remain unchanged) ...
    # ... (start, stop, _data_fetch_loop, fetch_all_live_data, etc.) ...
    
    def get_current_prices(self) -> Dict[str, float]:
        """Récupère les prix actuels via CoinLore API (enhanced)"""
        # ... (Method content unchanged) ...
        return {}
        
class CryptoTradingSimulator:
    """Simulateur de trading de cryptomonnaies amélioré et parallèle"""
    
    def __init__(self, initial_capital, output_dir, log_file):
        self.starting_cash = initial_capital
        self.capital_key = f"portfolio_logs_{initial_capital}"
        self.output_dir = output_dir
        self.log_file = os.path.join(self.output_dir, log_file)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = self.setup_logging()
        self.enhanced_log('INFO', f"--- Initialisation du simulateur pour {self.capital_key} ---")
        
        self.transaction_cost = 0.001
        self.min_cash_pct = 0.05
        self.circuit_breaker_pct = 0.20
        
        self.alerts = {
            'large_drawdown': False,
            'high_volatility': False,
            'api_error': False,
            'performance_degradation': False
        }
        self.circuit_breaker_active = False
        self.circuit_breaker_trigger_time: Optional[datetime] = None
        self.performance_metrics = {'trades_executed': 0, 'api_calls': 0}
        
        self.params = {
            # ... (Params dictionary unchanged) ...
             "MeanReversion": { "period": 20, "buy_threshold": 0.99, "sell_threshold": 1.01, "max_position_pct": 0.08, "take_profit_pct": 1.015, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "limit_buy_offset_pct": -0.003, "trailing_stop_pct": 0.02, "timeframes": ['1m', '5m'] }, 
             "MA_Enhanced": { "short_window": 5, "long_window": 20, "volatility_threshold": 0.0005, "max_position_pct": 0.12, "stop_loss_pct": 0.06, "take_profit_pct": 0.15, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "timeframes": ['1m', '5m'] }, 
             "Momentum_Enhanced": { "period": 1, "threshold": 0.3, "smoothing": 1, "max_position_pct": 0.9, "stop_loss_pct": 0.7, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "timeframes": ['1m'] }, 
             "MeanReversion_Pro": { "period": 15, "buy_threshold": 0.98, "sell_threshold": 1.02, "volatility_filter": 0.01, "max_position_pct": 0.1, "dynamic_stop_pct": 0.03, "take_profit_pct": 0.05, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "timeframes": ['1m', '5m'] }, 
             "MA_Momentum_Hybrid": { "ma_short": 8, "ma_long": 21, "momentum_period": 5, "momentum_threshold": 0.02, "max_position_pct": 0.1, "stop_loss_pct": 0.05, "take_profit_pct": 0.1, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "timeframes": ['1m', '5m'] }, 
             "Volatility_Regime_Adaptive": { "low_vol_period": 10, "vol_threshold": 0.005, "max_position_pct": 0.15, "stop_loss_pct": 0.05, "take_profit_pct": 0.1, "min_price_variation_pct": 0.00005, "confirmation_periods": 1, "timeframes": ['1m', '5m'] }
        }
        
        self.update_interval = 5
        
        self.data_fetcher = DataFetcher(
            coinlore_id_map={
                # ... (coinlore_id_map unchanged) ...
                'BTC-USD': '90', 'ETH-USD': '80', 'BNB-USD': '2710', 'SOL-USD': '48543', 'XRP-USD': '58', 'DOGE-USD': '2', 'ADA-USD': '257', 'TRX-USD': '2727', 'AVAX-USD': '44883', 'DOT-USD': '45031', 'LINK-USD': '2751', 'MATIC-USD': '33536', 'SHIB-USD': '49302', 'LTC-USD': '1', 'ICP-USD': '47333', 'BCH-USD': '2321', 'ETC-USD': '60', 'NEAR-USD': '46013', 'XLM-USD': '51', 'FIL-USD': '2280', 'ATOM-USD': '3334', 'INJ-USD': '44105', 'HBAR-USD': '39366', 'IMX-USD': '48043', 'XMR-USD': '29', 'CRO-USD': '2995', 'APT-USD': '51111', 'LDO-USD': '48239', 'VET-USD': '3000'
            },
            update_interval=self.update_interval
        )
        
        self.chart_generator = AdvancedChartGenerator(output_dir=self.output_dir)
        
        self.indicators_cache = {}
        self.price_validation_cache = {}
        self.cache_ttl = timedelta(minutes=1)
        self._last_cleanup_time = datetime.now()
        self._last_rebalance_time: Optional[datetime] = None
        
        # --- STATEFUL MODIFICATIONS ---
        self.strategy_states: Dict[str, Dict] = {}
        self.state_lock = threading.Lock()
        # --- END STATEFUL MODIFICATIONS ---

    # --- NEW METHOD: save_state ---
    def save_state(self, filepath: str):
        """Saves the complete simulation state to a JSON file."""
        self.enhanced_log('INFO', f"--- SAVING SIMULATION STATE TO {filepath} ---")
        
        try:
            # 1. Serialize historical dataframes
            serializable_history_dfs = {}
            if hasattr(self, 'data_fetcher') and hasattr(self.data_fetcher, 'historical_data_dfs'):
                 for coin, df in self.data_fetcher.historical_data_dfs.items():
                    # Ensure index is datetime and serializable
                    df.index = pd.to_datetime(df.index)
                    serializable_history_dfs[coin] = df.to_json(orient='split', date_format='iso')

            # 2. Serialize strategy states (handling datetimes)
            serializable_strategy_states = copy.deepcopy(self.strategy_states)
            for state in serializable_strategy_states.values():
                if 'portfolio_history' in state:
                    state['portfolio_history'] = [
                        {**p, 'timestamp': p['timestamp'].isoformat()} 
                        for p in state['portfolio_history'] if isinstance(p.get('timestamp'), datetime)
                    ]
                if 'trade_log' in state:
                    state['trade_log'] = [
                        {**t, 'timestamp': t['timestamp'].isoformat()} 
                        for t in state['trade_log'] if isinstance(t.get('timestamp'), datetime)
                    ]

            # 3. Create final state data object
            state_data = {
                'historical_data_dfs_json': serializable_history_dfs,
                'strategy_states': serializable_strategy_states,
                'alerts': self.alerts,
                'circuit_breaker_active': self.circuit_breaker_active,
                'circuit_breaker_trigger_time': self.circuit_breaker_trigger_time.isoformat() if self.circuit_breaker_trigger_time else None,
                'performance_metrics': self.performance_metrics,
                '_last_cleanup_time': self._last_cleanup_time.isoformat(),
                '_last_rebalance_time': self._last_rebalance_time.isoformat() if self._last_rebalance_time else None,
                'indicators_cache': {k: {**v, 'timestamp': v['timestamp'].isoformat()} for k, v in self.indicators_cache.items()},
                'price_validation_cache': self.price_validation_cache
            }
            
            # 4. Write to file
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=4)
            self.enhanced_log('INFO', f"--- SUCCESSFULLY SAVED STATE ---")
            
        except Exception as e:
            self.enhanced_log('CRITICAL', f"FAILED TO SAVE STATE: {e}")
            print(f"CRITICAL ERROR: FAILED TO SAVE STATE: {e}")

    # --- NEW METHOD: load_state ---
    def load_state(self, filepath: str) -> bool:
        """Loads the simulation state from a JSON file."""
        self.enhanced_log('INFO', f"--- LOADING SIMULATION STATE FROM {filepath} ---")
        if not os.path.exists(filepath):
            self.enhanced_log('WARNING', "State file not found. Starting a fresh simulation.")
            return False
            
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)

            # 1. Restore historical dataframes
            if 'historical_data_dfs_json' in state_data:
                loaded_dfs = {}
                for coin, df_json in state_data['historical_data_dfs_json'].items():
                    df = pd.read_json(df_json, orient='split')
                    df.index = pd.to_datetime(df.index)
                    loaded_dfs[coin] = df
                self.data_fetcher.historical_data_dfs = loaded_dfs
                self.enhanced_log('INFO', f"Restored {len(loaded_dfs)} historical price dataframes.")

            # 2. Restore strategy states (handling datetimes)
            self.strategy_states = state_data.get('strategy_states', {})
            for strategy, state in self.strategy_states.items():
                if 'portfolio_history' in state:
                    try:
                        state['portfolio_history'] = [
                            {**p, 'timestamp': datetime.fromisoformat(p['timestamp'])} 
                            for p in state['portfolio_history'] if isinstance(p.get('timestamp'), str)
                        ]
                    except Exception as e:
                        self.enhanced_log('WARNING', f"Could not parse portfolio_history timestamp for {strategy}: {e}")
                        state['portfolio_history'] = []
                if 'trade_log' in state:
                    try:
                        state['trade_log'] = [
                            {**t, 'timestamp': datetime.fromisoformat(t['timestamp'])} 
                            for t in state['trade_log'] if isinstance(t.get('timestamp'), str)
                        ]
                    except Exception as e:
                         self.enhanced_log('WARNING', f"Could not parse trade_log timestamp for {strategy}: {e}")
                         state['trade_log'] = []

            # 3. Restore class-level variables
            self.alerts = state_data.get('alerts', self.alerts)
            self.circuit_breaker_active = state_data.get('circuit_breaker_active', False)
            cb_time_str = state_data.get('circuit_breaker_trigger_time')
            self.circuit_breaker_trigger_time = datetime.fromisoformat(cb_time_str) if cb_time_str else None
            
            self.performance_metrics = state_data.get('performance_metrics', self.performance_metrics)
            
            lct_str = state_data.get('_last_cleanup_time', datetime.now().isoformat())
            self._last_cleanup_time = datetime.fromisoformat(lct_str)
            
            lrb_str = state_data.get('_last_rebalance_time')
            self._last_rebalance_time = datetime.fromisoformat(lrb_str) if lrb_str else None

            # Restore caches (handle datetime conversion)
            try:
                self.indicators_cache = {
                    k: {**v, 'timestamp': datetime.fromisoformat(v['timestamp'])} 
                    for k, v in state_data.get('indicators_cache', {}).items()
                }
                self.price_validation_cache = state_data.get('price_validation_cache', {})
            except Exception as e:
                 self.enhanced_log('WARNING', f"Could not parse cache timestamps: {e}")
                 self.indicators_cache = {}
                 self.price_validation_cache = {}


            self.enhanced_log('INFO', f"--- SUCCESSFULLY LOADED STATE. {len(self.strategy_states)} strategy states restored. ---")
            return True
            
        except Exception as e:
            self.enhanced_log('CRITICAL', f"FAILED TO LOAD STATE: {e}. Starting fresh.")
            print(f"CRITICAL ERROR: FAILED TO LOAD STATE: {e}. Starting fresh.")
            self.strategy_states = {} # Ensure state is clean
            self.data_fetcher.historical_data_dfs = {k: pd.DataFrame(columns=list(self.data_fetcher.coinlore_id_map.keys())) for k in self.data_fetcher.historical_data_dfs} # Clear history
            return False

    def setup_logging(self):
        # ... (Method content unchanged) ...
        return logger
        
    def enhanced_log(self, level, message, strategy=None, coin=None):
        # ... (Method content unchanged) ...
        
    def memory_optimization_cleanup(self):
        # ... (Method content unchanged) ...

    def validate_price_enhanced(self, price: float, previous_price: float = None, coin: str = None, timeframe: str = '1m') -> Tuple[bool, str]:
        # ... (Method content unchanged) ...
        return False, "Price validation error"
        
    def get_indicator(self, strategy: str, coin: str, prices_df: pd.DataFrame, indicator: str, params: Dict, timeframe: str):
        # ... (Method content unchanged) ...
        return None

    def calculate_enhanced_risk_metrics(self, portfolio_history: List[Dict]) -> Dict:
        # ... (Method content unchanged) ...
        return {}

    def calculate_sharpe_ratio(self, portfolio_history: List[Dict], risk_free_rate=0) -> float:
        # ... (Method content unchanged) ...
        return 0.0

    def calculate_sortino_ratio(self, portfolio_history: List[Dict], risk_free_rate=0) -> float:
        # ... (Method content unchanged) ...
        return 0.0
        
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        # ... (Method content unchanged) ...
        return 0.0
        
    def calculate_calmar_ratio(self, total_return: float, portfolio_values: List[float]) -> float:
        # ... (Method content unchanged) ...
        return 0.0
        
    def calculate_var(self, returns: pd.Series, confidence_level=0.95) -> float:
        # ... (Method content unchanged) ...
        return 0.0
        
    def calculate_expected_shortfall(self, returns: pd.Series, confidence_level=0.95) -> float:
        # ... (Method content unchanged) ...
        return 0.0
        
    def calculate_ulcer_index(self, portfolio_values: List[float]) -> float:
        # ... (Method content unchanged) ...
        return 0.0

    def calculate_win_rate(self, trade_log: List[Dict]) -> float:
        # ... (Method content unchanged) ...
        return 0.0
        
    def calculate_kelly_position_size(self, strategy: str, cash: float, trade_log: list) -> float:
        # ... (Method content unchanged) ...
        return cash * 0.05 # default
        
    def calculate_dynamic_position_size(self, strategy: str, cash: float, volatility: float) -> float:
        # ... (Method content unchanged) ...
        return min(0, cash * 0.01) # default
        
    def calculate_risk_parity_position_size(self, strategy: str, cash: float, coin: str) -> float:
        # ... (Method content unchanged) ...
        return cash * 0.1 * 0.7 # default
        
    def enhanced_position_sizing(self, strategy: str, cash: float, trade_log: list, current_volatility: float, coin: str = None) -> float:
        # ... (Method content unchanged) ...
        return 0.0
        
    def portfolio_rebalancing(self, holdings: Dict, prices: Dict, cash: float, strategy: str) -> Tuple[Dict, float, List[Dict]]:
        # ... (Method content unchanged) ...
        return holdings, cash, []

    def enhanced_order_management(self, orders: List[Order], prices: Dict, strategy: str) -> Tuple[List[Order], List[Dict]]:
        # ... (Method content unchanged) ...
        return [], []
        
    def run_enhanced_single_strategy(self, coins, strategy, duration_minutes=2, lookback_period=30):
        """Enhanced single strategy runner with all new features"""
        
        # --- STATEFUL MODIFICATION: LOAD STATE ---
        if strategy in self.strategy_states:
            self.enhanced_log('INFO', f"Loading saved state for strategy: {strategy}", strategy)
            state = self.strategy_states[strategy]
            cash = state.get('cash', self.starting_cash)
            holdings = state.get('holdings', {})
            # Clear open orders on resume; managing live orders across sessions is too complex/risky
            open_orders = {} 
            trade_log = state.get('trade_log', [])
            portfolio_history = state.get('portfolio_history', [])
            entry_prices = state.get('entry_prices', {})
            peak_portfolio_value = state.get('peak_portfolio_value', self.starting_cash)
            
            # Ensure portfolio_history isn't empty
            if not portfolio_history:
                 portfolio_history = [{'timestamp': datetime.now(), 'value': cash, 'cash': cash, 'holdings_value': 0}]
            
        else:
            self.enhanced_log('INFO', f"No saved state found. Starting fresh for: {strategy}", strategy)
            cash = self.starting_cash
            holdings = {}
            open_orders = {}
            trade_log = []
            # Start portfolio history with the initial state
            portfolio_history = [{
                'timestamp': datetime.now(), 
                'value': cash, 
                'cash': cash, 
                'holdings_value': 0
            }]
            entry_prices = {}
            peak_portfolio_value = cash
        # --- END STATEFUL MODIFICATION ---
        
        start_time = datetime.now()
        previous_prices = {}
        
        if not self.data_fetcher.is_running:
            self.enhanced_log('INFO', "Data fetcher not running. Starting it.", strategy)
            self.data_fetcher.start()
            time.sleep(10) # Give it time to fetch initial data

        try:
            while (datetime.now() - start_time).total_seconds() < duration_minutes * 60:
                if self.circuit_breaker_active:
                    # ... (Circuit breaker logic unchanged) ...
                    if (datetime.now() - self.circuit_breaker_trigger_time).total_seconds() > 300:
                         self.circuit_breaker_active = False
                         self.enhanced_log('INFO', "Circuit breaker reset.", strategy)
                    time.sleep(10)
                    continue

                prices = self.data_fetcher.get_live_data()
                if not prices:
                    self.enhanced_log('WARNING', "No prices data available. Skipping loop.", strategy)
                    time.sleep(self.update_interval)
                    continue
                
                self.performance_metrics['api_calls'] += 1
                
                # ... (Order management logic unchanged) ...
                open_orders_list = [order for coin_orders in open_orders.values() for order in coin_orders]
                open_orders_list, executed_trades = self.enhanced_order_management(open_orders_list, prices, strategy)
                
                # ... (Trade execution logic unchanged) ...
                for execution in executed_trades:
                    # ... (Logic for handling BUY/SELL) ...
                    pass # Placeholder for the large block of trade logic

                # ... (Portfolio rebalancing logic unchanged) ...
                holdings, cash, rebalance_trades = self.portfolio_rebalancing(holdings, prices, cash, strategy)
                trade_log.extend(rebalance_trades)
                
                # ... (Signal generation and new order logic unchanged) ...
                for coin in coins:
                    if coin not in prices:
                        continue
                        
                    # ... (Logic to get history, validate prices, generate signals) ...
                    history = self.data_fetcher.get_historical_data(coin, '1m', lookback_period)
                    if history is None or history.empty:
                        continue
                    
                    signal = self.generate_enhanced_signals(strategy, coin, history, prices)
                    
                    # ... (Logic to place orders based on signal) ...
                    if signal == 1 and f"{coin}_BUY" not in [o.id for o in open_orders.get(coin, [])]:
                        # ... (Position sizing and BUY order creation) ...
                        pass
                    elif signal == -1 and holdings.get(coin, 0) > 0 and f"{coin}_SELL" not in [o.id for o in open_orders.get(coin, [])]:
                        # ... (SELL order creation and OCO logic) ...
                        pass
                
                # ... (Portfolio history and drawdown logic unchanged) ...
                port_value = cash + sum(holdings.get(c, 0) * prices.get(c, 0) for c in coins)
                portfolio_history.append({
                    'timestamp': datetime.now(),
                    'value': port_value,
                    'cash': cash,
                    'holdings_value': port_value - cash
                })
                
                peak_portfolio_value = max(peak_portfolio_value, port_value)
                current_drawdown = (peak_portfolio_value - port_value) / peak_portfolio_value
                if current_drawdown > self.max_drawdown_pct:
                    # ... (Drawdown alert logic) ...
                    pass
                
                previous_prices = prices.copy()
                time.sleep(self.update_interval)

        except Exception as e:
            self.enhanced_log('ERROR', f"Error in enhanced trading loop: {e}", strategy)
            time.sleep(self.update_interval)
            
        # --- STATEFUL MODIFICATION: SAVE STATE ---
        # Ensure datetimes are valid before saving
        final_portfolio_history = [p for p in portfolio_history if isinstance(p.get('timestamp'), datetime)]
        final_trade_log = [t for t in trade_log if isinstance(t.get('timestamp'), datetime)]
        
        final_state = {
            'cash': cash,
            'holdings': holdings,
            'open_orders': {},  # Do not save open orders across sessions
            'trade_log': final_trade_log,
            'portfolio_history': final_portfolio_history,
            'entry_prices': entry_prices,
            'peak_portfolio_value': peak_portfolio_value
        }
        with self.state_lock:
            self.strategy_states[strategy] = final_state
        # --- END STATEFUL MODIFICATION ---

        # Final calculations
        final_value = portfolio_history[-1]['value'] if portfolio_history else self.starting_cash
        # Total return is calculated from the *original* starting cash, which is correct
        total_return = (final_value - self.starting_cash) / self.starting_cash
        
        # Risk metrics use the *full* history, which is also correct
        risk_metrics = self.calculate_enhanced_risk_metrics(portfolio_history)
        
        results = {
            'Return': total_return * 100,
            'Final Value': final_value,
            'Trades': len([t for t in trade_log if t['action'] == 'BUY']),
            'Win Rate': self.calculate_win_rate(trade_log),
            'Sharpe Ratio': risk_metrics.get('sharpe_ratio', 0),
            'Sortino Ratio': risk_metrics.get('sortino_ratio', 0),
            'Max Drawdown (%)': risk_metrics.get('max_drawdown', 0) * 100,
            'Volatility': risk_metrics.get('volatility', 0),
            'Calmar Ratio': risk_metrics.get('calmar_ratio', 0),
            'Ulcer Index': risk_metrics.get('ulcer_index', 0),
        }

        return {
            'results': results,
            'trade_log': trade_log,
            'portfolio_history': portfolio_history
        }

    def generate_enhanced_signals(self, strategy: str, coin: str, history: pd.Series, prices: Dict) -> int:
        # ... (Method content unchanged) ...
        return 0
        
    def run_parallel_strategies(self, coins, strategies, duration_minutes=2, lookback_period=30):
        # ... (Method content unchanged) ...
        all_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            future_to_strategy = {
                executor.submit(self.run_enhanced_single_strategy, coins, strategy, duration_minutes, lookback_period): strategy
                for strategy in strategies
            }
            
            for future in concurrent.futures.as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result()
                    all_results[strategy] = result
                    self.enhanced_log('INFO', f"Strategy {strategy} completed.", strategy)
                except Exception as exc:
                    self.enhanced_log('CRITICAL', f"Strategy {strategy} generated an exception: {exc}", strategy)

        self.data_fetcher.stop() # Stop the data fetcher when all strats are done
        return all_results
        
    def generate_performance_report(self, all_results: Dict) -> Dict:
        # ... (Method content unchanged) ...
        return {'summary': {}, 'details': {}}
        
    def get_current_prices(self) -> Dict:
        # ... (Method content unchanged) ...
        return {}

    def real_time_monitoring_dashboard(self, all_results: Dict, current_prices: Dict):
        # ... (Method content unchanged) ...
        
    def export_results(self, all_results: Dict):
        # ... (Method content unchanged) ...
        
    def generate_advanced_charts(self, all_results: Dict, performance_report: Dict) -> Dict:
        # ... (Method content unchanged) ...
        return {}

# --- MODIFIED MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    # --- STATEFUL MODIFICATION ---
    # This value must match the one in the YML matrix (e.g., 10, 100, 1000)
    CAPITAL = 10 
    OUTPUT_DIR = f"portfolio_logs_{CAPITAL}"
    STATE_FILE = os.path.join(OUTPUT_DIR, "simulation_state.json")
    
    # Ensure output directory exists (needed for state file)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "charts"), exist_ok=True)

    simulator = CryptoTradingSimulator(
        initial_capital=CAPITAL, 
        output_dir=OUTPUT_DIR,
        log_file="enhanced_simulation.log"
    )

    # --- LOAD STATE ---
    try:
        is_resumed_run = simulator.load_state(STATE_FILE)
        if is_resumed_run:
            print(f"RESUMING simulation for capital {CAPITAL} from {STATE_FILE}...")
        else:
            print(f"STARTING NEW simulation for capital {CAPITAL}...")
    except Exception as e:
        print(f"Error loading state: {e}. Starting NEW simulation.")
    # --- END LOAD STATE ---
    
    # Wrap the main run in try...finally to ensure state is saved
    try:
        print("Starting enhanced cryptocurrency trading simulation with REAL CoinLore data...")
        
        all_coins = simulator.data_fetcher.get_top_coins(50)
        
        # This list is from your original script
        selected_strategies = [
            "MeanReversion", 
            "MA_Enhanced", 
            "Momentum_Enhanced", 
            "MeanReversion_Pro", 
            "MA_Momentum_Hybrid",
            "Volatility_Regime_Adaptive"
        ]

        print(f"Running optimized strategy set: {selected_strategies}")
        
        simulation_results = simulator.run_parallel_strategies(
            coins=all_coins,
            strategies=selected_strategies,
            duration_minutes=300,  # 5 hours
            lookback_period=30
        )

        # --- All original reporting/charting code follows ---
        performance_report = simulator.generate_performance_report(simulation_results)
        
        current_prices = simulator.get_current_prices()
        simulator.real_time_monitoring_dashboard(simulation_results, current_prices)
        
        simulator.export_results(simulation_results)
        
        print("\n🎯 PERFORMANCE REPORT SUMMARY:")
        print(f"Best Strategy: {performance_report['summary'].get('best_strategy', 'N/A')}")
        print(f"Best Sharpe: {performance_report['summary'].get('best_sharpe', 0):.2f}")
        print(f"Total Trades: {performance_report['summary'].get('total_trades', 0)}")
        print("📊 Génération des visualisations avancées...")
        
        chart_files = simulator.generate_advanced_charts(simulation_results, performance_report)

        print("\n🎨 GRAPHIQUES GÉNÉRÉS:")
        for chart_type, filepath in chart_files.items():
            if filepath:
                print(f"   ✅ {chart_type}: {filepath}")
        
        print(f"\n📁 Tous les graphiques sont sauvegardés dans: {simulator.output_dir}/charts/")
        print("--- Simulation chunk complete. ---")

    finally:
        # --- SAVE STATE ---
        print("\n--- Main simulation block finished. Saving final state. ---")
        try:
            simulator.save_state(STATE_FILE)
            print(f"--- State successfully saved to {STATE_FILE} ---")
        except Exception as e:
            print(f"CRITICAL ERROR: FAILED TO SAVE STATE: {e}")
            simulator.enhanced_log('CRITICAL', f"FAILED TO SAVE STATE: {e}")
        # --- END SAVE STATE ---
