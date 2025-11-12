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

matplotlib.use('Agg')  # Pour meilleures performances

class AdvancedChartGenerator:
    """Générateur de graphiques avancés avec visualisations interactives"""
    
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

    def create_comprehensive_dashboard(self, all_results: Dict, performance_report: Dict):
        """Crée un tableau de bord complet avec multiples visualisations"""
        try:
            fig = plt.figure(figsize=(20, 25))
            gs = GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.3)
            
            # 1. Performance des stratégies
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_strategy_performance(ax1, all_results)
            
            # 2. Métriques de risque
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_risk_metrics(ax2, all_results)
            
            # 3. Évolution de la valeur du portefeuille
            ax3 = fig.add_subplot(gs[1, :])
            self._plot_portfolio_evolution(ax3, all_results)
            
            # 4. Drawdowns
            ax4 = fig.add_subplot(gs[2, :2])
            self._plot_drawdown_analysis(ax4, all_results)
            
            # 5. Heatmap de corrélation des performances
            ax5 = fig.add_subplot(gs[2, 2:])
            self._plot_performance_heatmap(ax5, all_results)
            
            # 6. Distribution des rendements
            ax6 = fig.add_subplot(gs[3, :2])
            self._plot_return_distribution(ax6, all_results)
            
            # 7. Allocation optimale
            ax7 = fig.add_subplot(gs[3, 2:])
            self._plot_optimal_allocation(ax7, all_results)
            
            # 8. Métriques de trading
            ax8 = fig.add_subplot(gs[4, :])
            self._plot_trading_metrics(ax8, all_results)
            
            plt.suptitle('DASHBOARD COMPLET - ANALYSE DE PERFORMANCE DES STRATÉGIES', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Sauvegarde
            filename = os.path.join(self.output_dir, "charts", "comprehensive_dashboard.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filename
        except Exception as e:
            print(f"Erreur lors de la création du tableau de bord: {e}")
            return None

    def _plot_strategy_performance(self, ax, all_results: Dict):
        """Graphique de performance comparée des stratégies"""
        try:
            strategies = []
            returns = []
            sharpe_ratios = []
            
            for strategy, data in all_results.items():
                results = data.get('results', {})
                strategies.append(strategy)
                returns.append(results.get('Return', 0))
                sharpe_ratios.append(results.get('Sharpe Ratio', 0))
            
            if not strategies:
                ax.text(0.5, 0.5, 'Aucune donnée disponible', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('COMPARAISON DES PERFORMANCES DES STRATÉGIES', fontweight='bold')
                return
            
            # Double axe pour rendement et ratio de Sharpe
            x = np.arange(len(strategies))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, returns, width, label='Rendement (%)', 
                          color=self.colors['primary'], alpha=0.8)
            ax.set_ylabel('Rendement (%)', color=self.colors['primary'])
            ax.tick_params(axis='y', labelcolor=self.colors['primary'])
            
            ax2 = ax.twinx()
            bars2 = ax2.bar(x + width/2, sharpe_ratios, width, label='Sharpe Ratio', 
                           color=self.colors['secondary'], alpha=0.8)
            ax2.set_ylabel('Sharpe Ratio', color=self.colors['secondary'])
            ax2.tick_params(axis='y', labelcolor=self.colors['secondary'])
            
            # Améliorations visuelles
            ax.set_title('COMPARAISON DES PERFORMANCES DES STRATÉGIES', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=45, ha='right')
            
            # Ajout des valeurs sur les barres
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Légende combinée
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
        except Exception as e:
            print(f"Erreur dans _plot_strategy_performance: {e}")
            ax.text(0.5, 0.5, 'Erreur de visualisation', 
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_risk_metrics(self, ax, all_results: Dict):
        """Radar chart des métriques de risque"""
        try:
            metrics = ['Sharpe', 'Sortino', 'Win Rate', 'Calmar', 'Ulcer Index']
            strategies = list(all_results.keys())
            
            if not strategies:
                ax.text(0.5, 0.5, 'Aucune donnée disponible', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('ANALYSE DES MÉTRIQUES DE RISQUE', fontweight='bold')
                return
            
            # Préparation des données
            data = []
            valid_strategies = []
            
            for strategy in strategies:
                results = all_results[strategy].get('results', {})
                strategy_data = [
                    max(results.get('Sharpe Ratio', 0), 0),
                    max(results.get('Sortino Ratio', 0), 0),
                    results.get('Win Rate', 0) * 100,
                    min(max(results.get('Calmar Ratio', 0), 0), 10),
                    min(results.get('Ulcer Index', 0) * 100, 50)
                ]
                # Vérifier si les données sont valides
                if any(np.isnan(val) for val in strategy_data) or all(v == 0 for v in strategy_data):
                    continue
                data.append(strategy_data)
                valid_strategies.append(strategy)
            
            if not data:
                ax.text(0.5, 0.5, 'Données de risque insuffisantes', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('ANALYSE DES MÉTRIQUES DE RISQUE', fontweight='bold')
                return
            
            # Configuration du radar chart
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Fermer le cercle
            
            # Tracer chaque stratégie
            for i, (strategy, strategy_data) in enumerate(zip(valid_strategies, data)):
                values = strategy_data + [strategy_data[0]]  # Fermer le cercle
                ax.plot(angles, values, 'o-', linewidth=2, label=strategy, 
                       color=self.palette[i % len(self.palette)])
                ax.fill(angles, values, alpha=0.1, color=self.palette[i % len(self.palette)])
            
            # Set ylim only if data is valid
            max_val = 1
            if data:
                try:
                    calc_max = max([max(d) for d in data])
                    if np.isfinite(calc_max) and calc_max > 0:
                        max_val = calc_max * 1.2
                    elif calc_max == 0:
                        max_val = 1
                except Exception:
                    max_val = 1 # Default on error
            
            ax.set_ylim(0, max_val)
            
            # Configuration des axes
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_title('ANALYSE DES MÉTRIQUES DE RISQUE\n(Radar Chart)', fontweight='bold')
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
        except Exception as e:
            print(f"Erreur dans _plot_risk_metrics: {e}")
            ax.text(0.5, 0.5, 'Erreur de visualisation', 
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_portfolio_evolution(self, ax, all_results: Dict):
        """Évolution temporelle de la valeur du portefeuille"""
        try:
            has_data = False
            for i, (strategy, data) in enumerate(all_results.items()):
                portfolio_history = data.get('portfolio_history', [])
                if portfolio_history and len(portfolio_history) > 1:
                    timestamps = [p['timestamp'] for p in portfolio_history if 'timestamp' in p]
                    values = [p['value'] for p in portfolio_history if 'value' in p]
                    
                    if len(timestamps) == len(values) and len(values) > 1:
                        ax.plot(timestamps, values, label=strategy, 
                               color=self.palette[i % len(self.palette)], linewidth=2)
                        has_data = True
                        
            if not has_data:
                ax.text(0.5, 0.5, 'Aucune donnée temporelle disponible', 
                       ha='center', va='center', transform=ax.transAxes)

            ax.set_title('ÉVOLUTION DE LA VALEUR DU PORTEFEUILLE', fontweight='bold')
            ax.set_xlabel('Temps')
            ax.set_ylabel('Valeur du Portefeuille ($)')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            
        except Exception as e:
            print(f"Erreur dans _plot_portfolio_evolution: {e}")
            ax.text(0.5, 0.5, 'Erreur de visualisation', 
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_drawdown_analysis(self, ax, all_results: Dict):
        """Analyse des drawdowns"""
        try:
            has_data = False
            for i, (strategy, data) in enumerate(all_results.items()):
                portfolio_history = data.get('portfolio_history', [])
                if portfolio_history and len(portfolio_history) > 1:
                    values = [p['value'] for p in portfolio_history if 'value' in p]
                    
                    if len(values) > 1:
                        peak = np.maximum.accumulate(values)
                        drawdown = (values - peak) / peak * 100
                        
                        # Plot line
                        ax.plot(drawdown, label=f'Drawdown {strategy}', 
                               color=self.palette[i % len(self.palette)], alpha=0.7)
                        
                        # Fill for each strategy inside the loop
                        ax.fill_between(range(len(drawdown)), drawdown, 0, 
                                       color=self.palette[i % len(self.palette)], alpha=0.1)
                        has_data = True

            if not has_data:
                ax.text(0.5, 0.5, 'Aucune donnée disponible', 
                       ha='center', va='center', transform=ax.transAxes)

            ax.set_title('ANALYSE DES DRAWDOWN (%)', fontweight='bold')
            ax.set_xlabel('Périodes')
            ax.set_ylabel('Drawdown (%)')
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Erreur dans _plot_drawdown_analysis: {e}")
            ax.text(0.5, 0.5, 'Erreur de visualisation', 
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_performance_heatmap(self, ax, all_results: Dict):
        """Heatmap de corrélation des rendements des stratégies"""
        try:
            strategies = list(all_results.keys())
            portfolio_returns = []
            valid_strategies = []

            for strategy in strategies:
                portfolio_history = all_results[strategy].get('portfolio_history', [])
                if portfolio_history and len(portfolio_history) > 1:
                    values = [p['value'] for p in portfolio_history if 'value' in p]
                    if len(values) > 1:
                        returns = pd.Series(values).pct_change().dropna()
                        if not returns.empty:
                            portfolio_returns.append(returns.rename(strategy))
                            valid_strategies.append(strategy)

            if len(valid_strategies) < 2:
                ax.text(0.5, 0.5, 'Données insuffisantes\npour corrélation', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('MATRICE DE CORRÉLATION', fontweight='bold')
                return

            # Aligner les séries temporelles
            returns_df = pd.concat(portfolio_returns, axis=1).dropna()
            
            if returns_df.empty or len(returns_df) < 2:
                 ax.text(0.5, 0.5, 'Données alignées insuffisantes\npour corrélation', 
                       ha='center', va='center', transform=ax.transAxes)
                 ax.set_title('MATRICE DE CORRÉLATION', fontweight='bold')
                 return

            corr_matrix = returns_df.corr()
            
            sns.heatmap(corr_matrix, ax=ax, annot=True, cmap='RdYlBu_r', 
                        vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
            
            ax.set_title('MATRICE DE CORRÉLATION DES RENDEMENTS', fontweight='bold')
            ax.set_xticklabels(valid_strategies, rotation=45, ha='right')
            ax.set_yticklabels(valid_strategies, rotation=0)
            
        except Exception as e:
            print(f"Erreur dans _plot_performance_heatmap: {e}")
            ax.text(0.5, 0.5, 'Erreur de visualisation', 
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_return_distribution(self, ax, all_results: Dict):
        """Distribution des rendements"""
        try:
            returns_data = []
            strategies = []
            
            for strategy, data in all_results.items():
                portfolio_history = data.get('portfolio_history', [])
                if portfolio_history and len(portfolio_history) > 1:
                    values = [p['value'] for p in portfolio_history if 'value' in p]
                    if len(values) > 1:
                        daily_returns = pd.Series(values).pct_change().dropna() * 100
                        if not daily_returns.empty:
                            returns_data.append(daily_returns)
                            strategies.append(strategy)

            if not returns_data:
                ax.text(0.5, 0.5, 'Aucune donnée de rendement', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('DISTRIBUTION DES RENDEMENTS', fontweight='bold')
                return

            # Boxplot
            # Changed 'labels' to 'tick_labels'
            bplot = ax.boxplot(returns_data, patch_artist=True, tick_labels=strategies, 
                             showfliers=False) # Sans outliers pour clarté
            
            for patch, color in zip(bplot['boxes'], self.palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
                
            ax.set_title('DISTRIBUTION DES RENDEMENTS (Boxplot)', fontweight='bold')
            ax.set_ylabel('Rendement Périodique (%)')
            ax.grid(True, axis='y', alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        except Exception as e:
            print(f"Erreur dans _plot_return_distribution: {e}")
            ax.text(0.5, 0.5, 'Erreur de visualisation', 
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_optimal_allocation(self, ax, all_results: Dict):
        """Frontière efficiente et allocation optimale"""
        try:
            returns = []
            volatilities = []
            sharpes = []
            valid_strategies = []

            for strategy, data in all_results.items():
                results = data.get('results', {})
                ret = results.get('Return', 0)
                vol = results.get('Volatility', 0)
                sharpe = results.get('Sharpe Ratio', 0)
                
                if vol > 0: # Ignorer les stratégies sans volatilité
                    returns.append(ret)
                    volatilities.append(vol * 100) # En %
                    sharpes.append(sharpe)
                    valid_strategies.append(strategy)
            
            if len(valid_strategies) > 1:
                # Normalisation des Sharpes pour la taille
                sizes = [max(10, (s+1)*100) for s in sharpes]
                
                # Scatter plot
                scatter = ax.scatter(volatilities, returns, c=sharpes, 
                                   cmap='viridis', s=sizes, alpha=0.7,
                                   edgecolors='black', linewidths=0.5)
                
                # Annotations
                for i, strategy in enumerate(valid_strategies):
                    ax.annotate(strategy, (volatilities[i], returns[i]), 
                                xytext=(10, 10), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

                ax.set_title('FRONTIÈRE EFFICIENTE - RISQUE vs RENDEMENT', fontweight='bold')
                ax.set_xlabel('Volatilité (%)')
                ax.set_ylabel('Rendement (%)')
                ax.grid(True, alpha=0.3)
                
                # Barre de couleur
                plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
            
            else:
                ax.text(0.5, 0.5, 'Données insuffisantes\npour la frontière efficiente', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('ALLOCATION OPTIMALE', fontweight='bold')
                
        except Exception as e:
            print(f"Erreur dans _plot_optimal_allocation: {e}")
            ax.text(0.5, 0.5, 'Erreur de visualisation', 
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_trading_metrics(self, ax, all_results: Dict):
        """Métriques de trading (Nb trades, Win Rate, Max DD)"""
        try:
            strategies = []
            trades = []
            win_rates = []
            max_drawdowns = []

            for strategy, data in all_results.items():
                results = data.get('results', {})
                strategies.append(strategy)
                trades.append(results.get('Trades', 0))
                win_rates.append(results.get('Win Rate', 0) * 100)
                max_drawdowns.append(results.get('Max Drawdown (%)', 0))

            if not strategies:
                ax.text(0.5, 0.5, 'Aucune donnée de trading', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('MÉTRIQUES DE TRADING COMPARÉES', fontweight='bold')
                return

            # Graphique à barres groupées
            x = np.arange(len(strategies))
            width = 0.25
            
            bars1 = ax.bar(x - width, trades, width, label='Nombre de Trades', 
                          color=self.colors['primary'])
            bars2 = ax.bar(x, win_rates, width, label='Taux de Succès (%)', 
                          color=self.colors['success'])
            bars3 = ax.bar(x + width, max_drawdowns, width, label='Max Drawdown (%)', 
                          color=self.colors['danger'])
            
            ax.set_title('MÉTRIQUES DE TRADING COMPARÉES', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Ajouter les valeurs sur les barres
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=8)

        except Exception as e:
            print(f"Erreur dans _plot_trading_metrics: {e}")
            ax.text(0.5, 0.5, 'Erreur de visualisation', 
                   ha='center', va='center', transform=ax.transAxes)

    def create_interactive_dashboard(self, all_results: Dict):
        """Crée un tableau de bord interactif avec Plotly"""
        try:
            strategies = list(all_results.keys())
            if not strategies:
                return None
                
            fig = make_subplots(
                rows=3, cols=2,
                specs=[
                    [{'type': 'xy'}, {'type': 'xy'}],
                    [{'type': 'xy'}, {'type': 'polar'}],
                    [{'type': 'box'}, {'type': 'heatmap'}]
                ],
                subplot_titles=(
                    'Performance (Rendement vs Sharpe)',
                    'Évolution du Portefeuille',
                    'Analyse des Drawdowns (%)',
                    'Métriques de Risque (Radar)',
                    'Distribution des Rendements',
                    'Matrice de Corrélation'
                )
            )

            # 1. Performance (Rendement vs Sharpe)
            perf_data = []
            for strategy, data in all_results.items():
                results = data.get('results', {})
                perf_data.append({
                    'Strategy': strategy,
                    'Return': results.get('Return', 0),
                    'Sharpe': results.get('Sharpe Ratio', 0)
                })
            
            for i, data in enumerate(perf_data):
                fig.add_trace(
                    go.Bar(name=f"Return {data['Strategy']}", 
                           x=[data['Strategy']], y=[data['Return']], 
                           marker_color=self.palette[i % len(self.palette)]),
                    row=1, col=1
                )
            
            # 2. Évolution du Portefeuille
            for i, (strategy, data) in enumerate(all_results.items()):
                portfolio_history = all_results[strategy].get('portfolio_history', [])
                if portfolio_history:
                    timestamps = [p['timestamp'] for p in portfolio_history if 'timestamp' in p]
                    values = [p['value'] for p in portfolio_history if 'value' in p]
                    if len(timestamps) == len(values) and len(values) > 1:
                        fig.add_trace(
                            go.Scatter(x=timestamps, y=values, name=strategy, 
                                       line=dict(color=self.palette[i % len(self.palette)])),
                            row=1, col=2
                        )
            
            # 3. Drawdowns
            for i, strategy in enumerate(strategies):
                portfolio_history = all_results[strategy].get('portfolio_history', [])
                if portfolio_history:
                    values = [p['value'] for p in portfolio_history if 'value' in p]
                    if len(values) > 1:
                        peak = np.maximum.accumulate(values)
                        drawdown = (values - peak) / peak * 100
                        fig.add_trace(
                            go.Scatter(y=drawdown, name=f'Drawdown {strategy}', 
                                       line=dict(color=self.palette[i % len(self.palette)])),
                            row=2, col=1
                        )
            
            # 4. Radar Chart (Métriques de Risque)
            metrics = ['Sharpe', 'Sortino', 'Win Rate', 'Calmar', 'Ulcer Index']
            for i, strategy in enumerate(strategies):
                results = all_results[strategy].get('results', {})
                values = [
                    max(results.get('Sharpe Ratio', 0), 0),
                    max(results.get('Sortino Ratio', 0), 0),
                    results.get('Win Rate', 0),
                    min(max(results.get('Calmar Ratio', 0), 0), 5),
                    min(results.get('Ulcer Index', 0), 2)
                ]
                fig.add_trace(
                    go.Scatterpolar(
                        r=values + [values[0]],
                        theta=metrics + [metrics[0]],
                        fill='toself',
                        name=strategy,
                        marker=dict(color=self.palette[i % len(self.palette)])
                    ),
                    row=2, col=2
                )
            
            # 5. Distribution des Rendements
            for i, (strategy, data) in enumerate(all_results.items()):
                portfolio_history = all_results[strategy].get('portfolio_history', [])
                if portfolio_history:
                    values = [p['value'] for p in portfolio_history if 'value' in p]
                    if len(values) > 1:
                        daily_returns = pd.Series(values).pct_change().dropna() * 100
                        if not daily_returns.empty:
                            fig.add_trace(
                                go.Box(y=daily_returns, name=strategy, 
                                       marker_color=self.palette[i % len(self.palette)]),
                                row=3, col=1
                            )

            # 6. Heatmap de corrélation
            if len(strategies) > 1:
                portfolio_returns = []
                valid_strategies = []
                for strategy in strategies:
                    portfolio_history = all_results[strategy].get('portfolio_history', [])
                    if portfolio_history:
                        values = [p['value'] for p in portfolio_history if 'value' in p]
                        if len(values) > 1:
                            returns = pd.Series(values).pct_change().dropna()
                            if not returns.empty:
                                portfolio_returns.append(returns.rename(strategy))
                                valid_strategies.append(strategy)
                
                if len(valid_strategies) > 1:
                    returns_df = pd.concat(portfolio_returns, axis=1).dropna()
                    if not returns_df.empty:
                        corr_matrix = returns_df.corr()
                        
                        # Replaced 'annot=True' with 'text' and 'texttemplate'
                        fig.add_trace(
                            go.Heatmap(z=corr_matrix.values, x=valid_strategies, y=valid_strategies, 
                                       colorscale='RdYlBu_r', zmin=-1, zmax=1,
                                       text=corr_matrix.values,
                                       texttemplate="%{text:.2f}"
                                       ),
                            row=3, col=2
                        )
            
            fig.update_layout(
                height=1600, 
                title_text="DASHBOARD INTERACTIF - ANALYSE DES STRATÉGIES", 
                showlegend=True
            )

            # Sauvegarde
            filename = os.path.join(self.output_dir, "charts", "interactive_dashboard.html")
            fig.write_html(filename)
            return filename
        
        except Exception as e:
            print(f"Erreur lors de la création du dashboard interactif: {e}")
            return None

    def create_animated_performance_chart(self, all_results: Dict):
        """Crée un graphique animé de la performance"""
        try:
            frames = []
            strategies = list(all_results.keys())
            valid_strategies = []

            # Préparer les données valides
            for strategy in strategies:
                portfolio_history = all_results[strategy].get('portfolio_history', [])
                if portfolio_history and len(portfolio_history) > 1:
                    valid_strategies.append(strategy)
            
            if not valid_strategies:
                return None

            # Créer les frames d'animation
            all_data = []
            max_len = 0
            
            for i, strategy in enumerate(valid_strategies):
                portfolio_history = all_results[strategy].get('portfolio_history', [])
                if portfolio_history:
                    timestamps = [p['timestamp'] for p in portfolio_history if 'timestamp' in p]
                    values = [p['value'] for p in portfolio_history if 'value' in p]
                    
                    if len(timestamps) == len(values) and len(values) > 1:
                        max_len = max(max_len, len(values))
                        df = pd.DataFrame({
                            'Timestamp': timestamps,
                            'Value': values,
                            'Strategy': strategy,
                            'Color': self.palette[i % len(self.palette)]
                        })
                        all_data.append(df)
            
            if not all_data:
                return None
                
            combined_df = pd.concat(all_data).sort_values(by='Timestamp')
            
            # Ajouter une frame_id (basée sur le temps)
            combined_df['Frame'] = pd.to_datetime(combined_df['Timestamp']).astype(np.int64) // 10**9
            combined_df = combined_df.sort_values(by='Frame')

            fig = px.line(
                combined_df, 
                x="Timestamp", 
                y="Value", 
                color="Strategy",
                animation_frame="Frame",
                animation_group="Strategy",
                title="ÉVOLUTION ANIMÉE DE LA PERFORMANCE DU PORTEFEUILLE",
                color_discrete_map={s: self.palette[i % len(self.palette)] for i, s in enumerate(valid_strategies)}
            )
            
            fig.update_layout(
                xaxis_title="Temps",
                yaxis_title="Valeur du Portefeuille ($)"
            )
            
            # Sauvegarde

            filename = os.path.join(self.output_dir, "charts", "animated_performance.html")
            fig.write_html(filename)
            return filename

        except Exception as e:
            print(f"Erreur lors de la création du graphique animé: {e}")
            return None

    def plot_cumulative_returns(self, all_results: Dict):
        """Trace les rendements cumulés de toutes les stratégies"""
        try:
            plt.figure(figsize=(14, 7))
            
            for i, (strategy, data) in enumerate(all_results.items()):
                portfolio_history = data.get('portfolio_history', [])
                if portfolio_history and len(portfolio_history) > 1:
                    values = [p['value'] for p in portfolio_history if 'value' in p]
                    if len(values) > 1:
                        returns = pd.Series(values).pct_change()
                        cumulative_returns = (1 + returns).cumprod() - 1
                        plt.plot(cumulative_returns.index, cumulative_returns * 100, 
                                 label=strategy, color=self.palette[i % len(self.palette)])

            plt.title('RENDEMENTS CUMULÉS DES STRATÉGIES', fontweight='bold', fontsize=14)
            plt.xlabel('Périodes')
            plt.ylabel('Rendement Cumulé (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            filename = os.path.join(self.output_dir, "charts", "cumulative_returns.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            return filename
        
        except Exception as e:
            print(f"Erreur dans plot_cumulative_returns: {e}")
            return None


class OrderType(Enum):
    MARKET_BUY = "MARKET_BUY"
    MARKET_SELL = "MARKET_SELL"
    LIMIT_BUY = "LIMIT_BUY"
    LIMIT_SELL = "LIMIT_SELL"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    OCO = "OCO"  # One-Cancels-Other

@dataclass
class Order:
    type: OrderType
    coin: str
    price: float
    amount: float
    timestamp: datetime
    expiration: datetime = None
    oco_group: str = None

class EnhancedParallelCryptoSimulatorCG:
    def __init__(self, starting_cash=1000.0, update_interval=5, max_history_points=1000, 
                 max_drawdown_pct=0.20, min_cash_pct=0.10, max_position_concentration=0.15):
        
        self.output_dir = "portfolio_logs_10"
        os.makedirs(self.output_dir, exist_ok=True)
        self.chart_generator = AdvancedChartGenerator(self.output_dir)
        
        self.starting_cash = starting_cash
        self.transaction_cost = 0.001
        self.update_interval = update_interval
        self.max_history_points = max_history_points
        self.max_drawdown_pct = max_drawdown_pct
        self.min_cash_pct = min_cash_pct
        self.max_position_concentration = max_position_concentration
        
        self.max_avg_correlation = 0.7
        self.slippage_pct = 0.001
        self.circuit_breaker_pct = 0.10
        self.order_expiration_minutes = 30
        self.rebalance_interval = timedelta(hours=1)
        
        self.live_data = {}
        self.is_running = False
        self.data_lock = threading.Lock()

        self.setup_enhanced_logging()
        
        self.indicators_cache = {}
        self.cache_ttl = timedelta(minutes=5)
        self.price_validation_cache = {}

        self.circuit_breaker_active = False
        self.circuit_breaker_trigger_time = None
        self.circuit_breaker_cooldown = timedelta(minutes=5)

        self.performance_metrics = {
            'api_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'trades_executed': 0
        }

        self.alerts = {
            'high_drawdown': False,
            'high_correlation': False,
            'data_quality_issue': False,
            'performance_degradation': False
        }

        # UPDATED: Only top 3 strategies + 2 enhanced + 1 new
        self.params = {
            # TOP 3 STRATEGIES FROM ANALYSIS
            "MeanReversion": {
                "period": 20, "buy_threshold": 0.99, "sell_threshold": 1.01, 
                "max_position_pct": 0.08, "take_profit_pct": 1.015, 
                "min_price_variation_pct": 0.00005, "confirmation_periods": 1, 
                "limit_buy_offset_pct": -0.003, "trailing_stop_pct": 0.02, 
                "timeframes": ['1m', '5m']
            },
            "MA_Enhanced": {
                "short_window": 5, "long_window": 20, "volatility_threshold": 0.0005, 
                "max_position_pct": 0.12, "stop_loss_pct": 0.06, "take_profit_pct": 0.15, 
                "min_price_variation_pct": 0.00005, "confirmation_periods": 1, 
                "timeframes": ['1m', '5m']
            },
            "Momentum_Enhanced": {
                "period": 1, "threshold": 0.3, "smoothing": 1, "max_position_pct": 0.9, 
                "stop_loss_pct": 0.7, "min_price_variation_pct": 0.00005, 
                "confirmation_periods": 1, "timeframes": ['1m']
            },
            
            # ENHANCED STRATEGIES (Learning from winners and avoiding losers)
            "MeanReversion_Pro": {
                "period": 15, "buy_threshold": 0.98, "sell_threshold": 1.02, 
                "volatility_filter": 0.002, "momentum_confirmation": True,
                "max_position_pct": 0.10, "stop_loss_pct": 0.04, "take_profit_pct": 0.12,
                "min_price_variation_pct": 0.00005, "confirmation_periods": 2,
                "timeframes": ['1m', '5m', '15m']
            },
            "MA_Momentum_Hybrid": {
                "ma_short": 8, "ma_long": 21, "momentum_period": 5, 
                "momentum_threshold": 0.02, "volume_confirmation": True,
                "max_position_pct": 0.15, "stop_loss_pct": 0.05, "take_profit_pct": 0.18,
                "min_price_variation_pct": 0.00005, "confirmation_periods": 1,
                "timeframes": ['1m', '5m']
            },
            
            # COMPLETELY NEW STRATEGY
            "Volatility_Regime_Adaptive": {
                "low_vol_period": 10, "high_vol_period": 5, "vol_threshold": 0.005,
                "trend_confirmation": True, "regime_smoothing": 3,
                "max_position_pct": 0.12, "stop_loss_pct": 0.03, "take_profit_pct": 0.10,
                "min_price_variation_pct": 0.00005, "confirmation_periods": 1,
                "timeframes": ['1m', '5m', '15m']
            }
        }

        self.coinlore_id_map = {
            "BTC-USD": "90", "ETH-USD": "80", "XRP-USD": "58", "BNB-USD": "2710",
            "SOL-USD": "48543", "DOGE-USD": "2", "ADA-USD": "257", "LINK-USD": "2751",
            "HBAR-USD": "48555", "AVAX-USD": "44883", "LTC-USD": "1", "SHIB-USD": "45088",
            "DOT-USD": "45219", "AAVE-USD": "46018", "NEAR-USD": "48563", "ICP-USD": "47311",
            "ATOM-USD": "33830", "SAND-USD": "45161", "AR-USD": "42441"
        }
        
        self.coinlore_reverse_map = {v: k for k, v in self.coinlore_id_map.items()}

        self.historical_data_dfs = {
            '1m': pd.DataFrame(columns=list(self.coinlore_id_map.keys())),
            '5m': pd.DataFrame(columns=list(self.coinlore_id_map.keys())),
            '15m': pd.DataFrame(columns=list(self.coinlore_id_map.keys()))
        }
        
        self._last_update_time = None
        self._last_timeframe_update = {tf: None for tf in self.historical_data_dfs.keys()}
        self._last_rebalance_time = datetime.now()
        self._last_cleanup_time = datetime.now()
        
    def setup_enhanced_logging(self):
        """Enhanced structured logging with rotation and levels"""
        self.logger = logging.getLogger('EnhancedCryptoSimulator')
        self.logger.setLevel(logging.INFO)
        
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        fh = RotatingFileHandler(
            os.path.join(self.output_dir, 'enhanced_simulation.log'),
            maxBytes=10*1024*1024,
            backupCount=5
        )
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(strategy)s] [%(coin)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def enhanced_log(self, level: str, message: str, strategy: str = "SYSTEM", 
                    coin: str = "GLOBAL", **kwargs):
        """Enhanced logging with performance tracking"""
        extra = {'strategy': strategy, 'coin': coin, **kwargs}
        
        if level.upper() == 'ERROR':
            self.performance_metrics['errors'] += 1
            
        if level.upper() == 'INFO':
            self.logger.info(message, extra=extra)
        elif level.upper() == 'WARNING':
            self.logger.warning(message, extra=extra)
        elif level.upper() == 'ERROR':
            self.logger.error(message, extra=extra)
        elif level.upper() == 'CRITICAL':
            self.logger.critical(message, extra=extra)

    def memory_optimization_cleanup(self):
        """Enhanced memory management with garbage collection"""
        current_time = datetime.now()
        if (current_time - self._last_cleanup_time).total_seconds() > 300:
            expired_keys = []
            for key, entry in list(self.indicators_cache.items()):
                if current_time - entry['timestamp'] > self.cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.indicators_cache[key]
                
            self.price_validation_cache.clear()
            gc.collect()
            self._last_cleanup_time = current_time
            self.enhanced_log('INFO', f"Memory cleanup completed. Cache entries: {len(self.indicators_cache)}")

    def validate_price_enhanced(self, price: float, previous_price: float = None, 
                              coin: str = None, timeframe: str = '1m') -> Tuple[bool, str]:
        """Enhanced price validation with multiple checks"""
        if price <= 0:
            return False, "Price must be positive"
            
        if np.isnan(price) or np.isinf(price):
            return False, "Price is NaN or infinite"
            
        if previous_price is not None and previous_price > 0:
            price_change = abs(price - previous_price) / previous_price
            
            if price_change > self.circuit_breaker_pct:
                self.enhanced_log('WARNING', 
                                f"Large price movement: {price_change:.2%}", 
                                coin=coin)
                # Allow large movements but log them, fail on extreme movements
                return price_change < 0.5, f"Large movement: {price_change:.2%}"
                
        return True, "Valid"

    def calculate_sharpe_ratio(self, portfolio_history: List) -> float:
        if not portfolio_history or len(portfolio_history) < 2: return 0.0
        try:
            values = [p.get('value', 0) for p in portfolio_history if 'value' in p]
            if len(values) < 2: return 0.0
            returns = pd.Series(values).pct_change().dropna()
            if returns.empty or returns.std() == 0: return 0.0
            return returns.mean() / returns.std() * np.sqrt(365 * 24 * 60)
        except Exception as e:
            self.enhanced_log('ERROR', f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def calculate_sortino_ratio(self, portfolio_history: List) -> float:
        if not portfolio_history or len(portfolio_history) < 2: return 0.0
        try:
            values = [p.get('value', 0) for p in portfolio_history if 'value' in p]
            if len(values) < 2: return 0.0
            returns = pd.Series(values).pct_change().dropna()
            if returns.empty: return 0.0
            downside_returns = returns[returns < 0]
            if downside_returns.empty: return float('inf')
            return returns.mean() / downside_returns.std() * np.sqrt(365 * 24 * 60)
        except Exception as e:
            self.enhanced_log('ERROR', f"Error calculating Sortino ratio: {e}")
            return 0.0

    def calculate_max_drawdown(self, values: List[float]) -> float:
        if not values or len(values) < 2: return 0.0
        try:
            peak = values[0]
            max_dd = 0.0
            for value in values[1:]:
                if value > peak: peak = value
                dd = (peak - value) / peak
                if dd > max_dd: max_dd = dd
            return max_dd
        except Exception as e:
            self.enhanced_log('ERROR', f"Error calculating max drawdown: {e}")
            return 0.0

    def calculate_enhanced_risk_metrics(self, portfolio_history: List) -> Dict:
        if not portfolio_history or len(portfolio_history) < 2: return {}
        try:
            values = [p.get('value', 0) for p in portfolio_history if 'value' in p]
            if len(values) < 2: return {}
            returns = pd.Series(values).pct_change().dropna()
            total_return = (values[-1] - values[0]) / values[0] if values[0] > 0 else 0
            volatility = returns.std() if not returns.empty else 0
            downside_returns = returns[returns < 0]
            upside_returns = returns[returns > 0]
            
            metrics = {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': self.calculate_sharpe_ratio(portfolio_history),
                'sortino_ratio': self.calculate_sortino_ratio(portfolio_history),
                'max_drawdown': self.calculate_max_drawdown(values),
                'calmar_ratio': self.calculate_calmar_ratio(total_return, values),
                'var_95': self.calculate_var(returns, 0.95),
                'var_99': self.calculate_var(returns, 0.99),
                'expected_shortfall_95': self.calculate_expected_shortfall(returns, 0.95),
                'win_rate': len(upside_returns) / len(returns) if len(returns) > 0 else 0,
                'profit_factor': abs(upside_returns.sum() / downside_returns.sum()) if len(downside_returns) > 0 and downside_returns.sum() != 0 else float('inf'),
                'ulcer_index': self.calculate_ulcer_index(values)
            }
            return metrics
        except Exception as e:
            self.enhanced_log('ERROR', f"Error calculating risk metrics: {e}")
            return {}

    def calculate_calmar_ratio(self, total_return: float, values: List[float]) -> float:
        max_dd = self.calculate_max_drawdown(values)
        return total_return / max_dd if max_dd > 0 else float('inf')

    def calculate_ulcer_index(self, values: List[float]) -> float:
        if len(values) < 2: return 0.0
        try:
            peak = values[0]
            drawdowns_sq = []
            for value in values[1:]:
                peak = max(peak, value)
                drawdown_pct = 100 * (value - peak) / peak
                drawdowns_sq.append(drawdown_pct ** 2)
            return np.sqrt(np.mean(drawdowns_sq))
        except Exception as e:
            self.enhanced_log('ERROR', f"Error calculating Ulcer Index: {e}")
            return 0.0

    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        if returns.empty: return 0.0
        return returns.quantile(1 - confidence_level)

    def calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        if returns.empty: return 0.0
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

    def calculate_kelly_position_size(self, strategy: str, cash: float, trade_log: List) -> float:
        if len(trade_log) < 10:
            return cash * self.params.get(strategy, {}).get("max_position_pct", 0.15)
        try:
            wins = [t['profit_pct'] for t in trade_log if t['profit_pct'] > 0]
            losses = [t['profit_pct'] for t in trade_log if t['profit_pct'] < 0]
            W = len(wins) / len(trade_log)
            if W == 0: return 0.0
            if W == 1: return cash * self.params.get(strategy, {}).get("max_position_pct", 0.15)
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            R = avg_win / avg_loss
            kelly_f = W - ((1 - W) / R)
            if kelly_f <= 0: return 0.0
            position_size = cash * kelly_f * 0.5
            return min(position_size, cash * self.params.get(strategy, {}).get("max_position_pct", 0.15))
        except Exception as e:
            self.enhanced_log('ERROR', f"Error in Kelly calculation: {e}", strategy)
            return cash * self.params.get(strategy, {}).get("max_position_pct", 0.15)

    def calculate_dynamic_position_size(self, strategy: str, cash: float, volatility: float) -> float:
        if volatility == 0: return cash * 0.01
        risk_per_trade_pct = 0.02
        stop_loss_pct = self.params.get(strategy, {}).get("stop_loss_pct", 0.05)
        base_size = (cash * risk_per_trade_pct) / stop_loss_pct
        vol_factor = 0.01 / volatility
        position_size = base_size * vol_factor
        return min(position_size, cash * self.params.get(strategy, {}).get("max_position_pct", 0.15))

    def calculate_risk_parity_position_size(self, strategy: str, cash: float, coin: str) -> float:
        base_pct = self.params.get(strategy, {}).get("max_position_pct", 0.15)
        return cash * base_pct * 0.7

    def enhanced_position_sizing(self, strategy: str, cash: float, trade_log: list, current_volatility: float, coin: str = None) -> float:
        reserved_cash = cash * self.min_cash_pct
        available_cash = cash - reserved_cash
        if available_cash <= 0: return 0.0
        
        kelly_size = self.calculate_kelly_position_size(strategy, available_cash, trade_log)
        vol_adjusted_size = self.calculate_dynamic_position_size(strategy, available_cash, current_volatility)
        risk_parity_size = self.calculate_risk_parity_position_size(strategy, available_cash, coin)
        
        avg_size = (kelly_size + vol_adjusted_size + risk_parity_size) / 3
        
        max_allowed = min(
            available_cash, 
            cash * self.params.get(strategy, {}).get("max_position_pct", 0.15),
            cash * self.max_position_concentration
        )
        return min(avg_size, max_allowed)

    def portfolio_risk_analysis(self, holdings: Dict, prices: Dict, strategy: str) -> Dict:
        analysis = {'concentration_risk': False, 'correlation_risk': False, 'avg_correlation': 0.0}
        try:
            total_value = sum(holdings.get(c, 0) * prices.get(c, 0) for c in holdings)
            if total_value == 0: return analysis
            
            for coin, amount in holdings.items():
                position_value = amount * prices.get(coin, 0)
                if (position_value / total_value) > self.max_position_concentration:
                    analysis['concentration_risk'] = True
                    self.enhanced_log('WARNING', f"High concentration in {coin}", strategy, coin)
                    break
            
            held_coins = [c for c in holdings if holdings[c] > 0]
            if len(held_coins) < 2: return analysis
                
            returns_list = []
            for coin in held_coins:
                if coin in self.historical_data_dfs['5m']:
                    returns = self.historical_data_dfs['5m'][coin].pct_change().dropna()
                    if not returns.empty:
                        returns_list.append(returns)
            
            if len(returns_list) < 2: return analysis
            returns_df = pd.concat(returns_list, axis=1).dropna()
            if returns_df.empty: return analysis

            corr_matrix = returns_df.corr()
            avg_correlation = corr_matrix.mean().mean()
            analysis['avg_correlation'] = avg_correlation
            
            if avg_correlation > self.max_avg_correlation:
                analysis['concentration_risk'] = True
                self.alerts['high_correlation'] = True
                self.enhanced_log('WARNING', f"High portfolio correlation: {avg_correlation:.3f}")
                
        except Exception as e:
            self.enhanced_log('ERROR', f"Error in correlation analysis: {e}")
        return analysis

    def portfolio_rebalancing(self, holdings: Dict, prices: Dict, cash: float, strategy: str) -> Tuple[Dict, float, List[Dict]]:
        rebalance_actions = []
        current_time = datetime.now()
        
        if (current_time - self._last_rebalance_time) < self.rebalance_interval:
            return holdings, cash, rebalance_actions
            
        try:
            total_value = cash + sum(holdings.get(c, 0) * prices.get(c, 0) for c in holdings)
            if total_value <= 0: return holdings, cash, rebalance_actions
            
            held_coins = [c for c in holdings if holdings[c] > 0]
            if not held_coins: return holdings, cash, rebalance_actions
                
            target_pct = 1.0 / len(held_coins)
            
            for coin in held_coins:
                current_value = holdings.get(coin, 0) * prices.get(coin, 0)
                current_pct = current_value / total_value
                target_value = total_value * target_pct
                
                if abs(current_pct - target_pct) > 0.05:
                    amount_diff = (target_value - current_value) / prices.get(coin, 0)
                    if amount_diff > 0:
                        cost = amount_diff * prices.get(coin, 0) * (1 + self.transaction_cost)
                        if cash >= cost:
                            holdings[coin] += amount_diff
                            cash -= cost
                            rebalance_actions.append({'action': 'BUY', 'coin': coin, 'amount': amount_diff})
                    else:
                        amount_to_sell = abs(amount_diff)
                        if holdings.get(coin, 0) >= amount_to_sell:
                            holdings[coin] -= amount_to_sell
                            cash += amount_to_sell * prices.get(coin, 0) * (1 - self.transaction_cost)
                            rebalance_actions.append({'action': 'SELL', 'coin': coin, 'amount': amount_to_sell})
                            
            self._last_rebalance_time = current_time
            if rebalance_actions:
                self.enhanced_log('INFO', f"Rebalancing executed for {strategy}", strategy)
                
        except Exception as e:
            self.enhanced_log('ERROR', f"Error during rebalancing: {e}", strategy)
        return holdings, cash, rebalance_actions

    def enhanced_order_management(self, orders: List[Order], prices: Dict, strategy: str) -> Tuple[List[Order], List[Dict]]:
        executed_orders = []
        orders_to_remove = []
        current_time = datetime.now()
        
        try:
            for i, order in enumerate(orders):
                if i in orders_to_remove: continue
                coin = order.coin
                current_price = prices.get(coin)
                if current_price is None: continue
                    
                if (order.expiration and current_time > order.expiration):
                    orders_to_remove.append(i)
                    self.enhanced_log('INFO', f"Order expired: {order.type} for {coin}", strategy, coin)
                    continue

                if order.oco_group:
                    oco_orders = [o for o in orders if o.oco_group == order.oco_group]
                    if len(oco_orders) > 1 and any(self.check_order_condition(o, current_price) for o in oco_orders if o != order):
                        orders_to_remove.extend(j for j, o in enumerate(orders) if o.oco_group == order.oco_group)
                        continue
                        
                if self.check_order_condition(order, current_price):
                    executed_orders.append({
                        'order': order,
                        'execution_price': current_price,
                        'timestamp': current_time
                    })
                    orders_to_remove.append(i)
            
            remaining_orders = [o for i, o in enumerate(orders) if i not in orders_to_remove]
            
        except Exception as e:
            self.enhanced_log('ERROR', f"Error in order management: {e}", strategy)
            return orders, []
            
        return remaining_orders, executed_orders

    def check_order_condition(self, order: Order, current_price: float) -> bool:
        if order.type == OrderType.LIMIT_BUY: return current_price <= order.price
        elif order.type == OrderType.LIMIT_SELL: return current_price >= order.price
        elif order.type == OrderType.STOP_LOSS: return current_price <= order.price
        elif order.type == OrderType.TAKE_PROFIT: return current_price >= order.price
        return False

    def start(self):
        """Start the live data update thread"""
        self.is_running = True
        self.enhanced_log('INFO', "Starting simulator...")
        self.data_thread = threading.Thread(target=self._data_fetch_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        self.enhanced_log('INFO', "Waiting for initial data fetch...")
        time.sleep(5) # Wait for first API call
        self.enhanced_log('INFO', "Initial data populated.")

    def stop(self):
        """Stop the simulator"""
        self.is_running = False
        if hasattr(self, 'data_thread'):
            self.data_thread.join()
        self.enhanced_log('INFO', "Simulator stopped.")

    def _data_fetch_loop(self):
        """Internal loop for fetching data"""
        while self.is_running:
            try:
                self.fetch_all_live_data()
                self.update_historical_dataframes()
                self.memory_optimization_cleanup()
            except Exception as e:
                self.enhanced_log('ERROR', f"Data fetch loop error: {e}")
            time.sleep(self.update_interval)

    def fetch_all_live_data(self):
        """Fetch live data for all coins"""
        prices = self.get_current_prices() 
        
        with self.data_lock:
            if prices: # Only update if data was successfully fetched
                self.live_data = prices
                self._last_update_time = datetime.now()
            
        return prices

    def update_historical_dataframes(self):
        """Update historical dataframes with the latest prices"""
        current_time = datetime.now()
        
        if not self.live_data:
            return
            
        with self.data_lock:
            try:
                # 1m update
                if (self._last_timeframe_update['1m'] is None or 
                    (current_time - self._last_timeframe_update['1m']).total_seconds() >= 60):
                    
                    new_row = self.live_data.copy()
                    new_row_filtered = {k: v for k, v in new_row.items() if k in self.historical_data_dfs['1m'].columns}
                    new_row_df = pd.DataFrame(new_row_filtered, index=[current_time])
                    
                    self.historical_data_dfs['1m'] = pd.concat([self.historical_data_dfs['1m'], new_row_df])
                    
                    if len(self.historical_data_dfs['1m']) > self.max_history_points:
                        self.historical_data_dfs['1m'] = self.historical_data_dfs['1m'].iloc[-self.max_history_points:]
                        
                    self._last_timeframe_update['1m'] = current_time
                
                # 5m update
                if (self._last_timeframe_update['5m'] is None or 
                    (current_time - self._last_timeframe_update['5m']).total_seconds() >= 300):
                    
                    new_row = self.live_data.copy()
                    new_row_filtered = {k: v for k, v in new_row.items() if k in self.historical_data_dfs['5m'].columns}
                    new_row_df = pd.DataFrame(new_row_filtered, index=[current_time])
                    self.historical_data_dfs['5m'] = pd.concat([self.historical_data_dfs['5m'], new_row_df])
                    self._last_timeframe_update['5m'] = current_time

                # 15m update
                if (self._last_timeframe_update['15m'] is None or 
                    (current_time - self._last_timeframe_update['15m']).total_seconds() >= 900):
                    
                    new_row = self.live_data.copy()
                    new_row_filtered = {k: v for k, v in new_row.items() if k in self.historical_data_dfs['15m'].columns}
                    new_row_df = pd.DataFrame(new_row_filtered, index=[current_time])
                    self.historical_data_dfs['15m'] = pd.concat([self.historical_data_dfs['15m'], new_row_df])
                    self._last_timeframe_update['15m'] = current_time
                    
            except Exception as e:
                self.enhanced_log('ERROR', f"Error updating historical data: {e}")

    # **MODIFIED FUNCTION**: Using your single-call logic
    def get_current_prices(self) -> Dict[str, float]:
        """
        Fetch current prices from CoinLore API for all configured coins
        in a single, targeted API call.
        """
        
        # ---
        # Step 1: Create the comma-separated ID string, as you suggested.
        # ---
        if not self.coinlore_id_map:
            self.enhanced_log('ERROR', "Coinlore ID map is empty.", "DATA_FETCH")
            return {}
            
        ids_string = ",".join(self.coinlore_id_map.values())
        api_url = f"https://api.coinlore.net/api/ticker/?id={ids_string}"
        
        prices = {}
        
        # ---
        # Step 2: Make the single API call
        # ---
        try:
            self.performance_metrics['api_calls'] += 1
            response = requests.get(api_url, timeout=5)
            response.raise_for_status()
            data = response.json()

            # ---
            # Step 3: Process the response (which is a list of coin objects)
            # ---
            if not isinstance(data, list):
                # Handle cases where only one ID is requested, and it might not be a list
                if isinstance(data, dict) and 'id' in data:
                     # This is a single-object response, wrap it in a list
                     data = [data]
                else:
                    self.enhanced_log('ERROR', f"CoinLore API did not return a list. Response: {data}", "DATA_FETCH")
                    return {}

            for coin in data:
                coin_id = coin.get('id')
                
                # Use the reverse map to find our internal symbol (e.g., "BTC-USD")
                if coin_id in self.coinlore_reverse_map:
                    internal_symbol = self.coinlore_reverse_map[coin_id]
                    try:
                        price = float(coin.get('price_usd'))
                        prices[internal_symbol] = price
                    except (TypeError, ValueError):
                        self.enhanced_log('WARNING', f"Invalid price data for coin ID {coin_id}", "DATA_FETCH", internal_symbol)
                else:
                    self.enhanced_log('WARNING', f"Received data for unmapped coin ID {coin_id}", "DATA_FETCH")

            # ---
            # Step 4: Final logging and return
            # ---
            if len(prices) != len(self.coinlore_id_map):
                self.enhanced_log('WARNING',
                                  f"Fetched {len(prices)}/{len(self.coinlore_id_map)} prices. "
                                  f"Missing: {[c for c in self.coinlore_id_map.keys() if c not in prices]}", 
                                  "DATA_FETCH")
            
            return prices

        except requests.exceptions.Timeout:
            self.enhanced_log('ERROR', "CoinLore API request timed out", "DATA_FETCH")
            return {}  # Return empty dict on timeout
        except requests.exceptions.RequestException as e:
            self.enhanced_log('ERROR', f"CoinLore API request failed: {e}", "DATA_FETCH")
            return {}  # Return empty dict on error
        except Exception as e:
            self.enhanced_log('ERROR', f"Error processing CoinLore data: {e}", "DATA_FETCH")
            return {}

    def run_enhanced_single_strategy(self, coins, strategy, duration_minutes=2, lookback_period=30):
        """Enhanced single strategy runner with all new features"""
        
        cash = self.starting_cash
        holdings = {coin: 0.0 for coin in coins}
        portfolio_history = []
        trade_log = []
        open_orders = {coin: [] for coin in coins}
        entry_prices = {coin: None for coin in coins}
        
        peak_portfolio_value = cash
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        self.enhanced_log('INFO', f"Starting enhanced strategy {strategy} with {len(coins)} coins", strategy)
        
        previous_prices = {}

        while datetime.now() < end_time and self.is_running:
            try:
                if self.circuit_breaker_active:
                    if (datetime.now() - self.circuit_breaker_trigger_time) > self.circuit_breaker_cooldown:
                        self.circuit_breaker_active = False
                        self.enhanced_log('INFO', "Circuit breaker cooldown finished.", strategy)
                    else:
                        time.sleep(self.update_interval)
                        continue

                with self.data_lock:
                    prices = self.live_data.copy()
                    
                if not prices:
                    self.enhanced_log('WARNING', "No live data available, sleeping...", strategy)
                    time.sleep(1)
                    continue

                valid_prices = {}
                for coin in coins:
                    price = prices.get(coin)
                    if price is None:
                        continue
                        
                    is_valid, reason = self.validate_price_enhanced(price, previous_prices.get(coin), coin)
                    if is_valid:
                        valid_prices[coin] = price
                    else:
                        self.enhanced_log('WARNING', f"Price validation failed for {coin}: {reason}", strategy, coin)
                
                if not valid_prices:
                    self.enhanced_log('WARNING', "No valid prices for strategy coins", strategy)
                    time.sleep(1)
                    continue
                
                prices = valid_prices
                
                all_open_orders = [order for coin_orders in open_orders.values() for order in coin_orders]
                remaining_orders, executed_orders = self.enhanced_order_management(all_open_orders, prices, strategy)
                
                open_orders = {coin: [] for coin in coins}
                for order in remaining_orders:
                    open_orders[order.coin].append(order)

                for execution in executed_orders:
                    order = execution['order']
                    exec_price = execution['execution_price']
                    coin = order.coin
                    
                    if order.type in [OrderType.LIMIT_BUY, OrderType.MARKET_BUY]:
                        cost = order.amount * exec_price * (1 + self.transaction_cost)
                        holdings[coin] += order.amount
                        cash -= cost
                        entry_prices[coin] = exec_price
                        trade_log.append({'action': 'BUY', 'coin': coin, 'amount': order.amount, 'price': exec_price, 'value': cost, 'timestamp': execution['timestamp']})
                        self.performance_metrics['trades_executed'] += 1
                            
                    elif order.type in [OrderType.LIMIT_SELL, OrderType.MARKET_SELL, OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
                        if holdings.get(coin, 0) >= order.amount:
                            holdings[coin] -= order.amount
                            revenue = order.amount * exec_price * (1 - self.transaction_cost)
                            cash += revenue
                            profit = (exec_price - entry_prices[coin]) * order.amount if entry_prices[coin] else 0
                            profit_pct = (exec_price - entry_prices[coin]) / entry_prices[coin] if entry_prices[coin] and entry_prices[coin] > 0 else 0
                            trade_log.append({'action': 'SELL', 'coin': coin, 'amount': order.amount, 'price': exec_price, 'value': revenue, 'profit': profit, 'profit_pct': profit_pct, 'timestamp': execution['timestamp']})
                            self.performance_metrics['trades_executed'] += 1
                            if holdings[coin] < 1e-9:
                                holdings[coin] = 0
                                entry_prices[coin] = None

                holdings, cash, rebalance_trades = self.portfolio_rebalancing(holdings, prices, cash, strategy)
                trade_log.extend(rebalance_trades)
                
                for coin in coins:
                    if coin not in prices:
                        continue
                        
                    current_price = prices[coin]
                    
                    if any(o.oco_group == f"{coin}_SLTP" for o in open_orders[coin]):
                        continue

                    strategy_params = self.params.get(strategy, {})
                    min_price_variation_pct = strategy_params.get("min_price_variation_pct", 0.0001)
                    
                    signal = self._calculate_signals_for_coin(
                        coin, prices, strategy, lookback_period, 
                        min_price_variation_pct, 
                        strategy_params.get("timeframes", ['1m', '5m'])
                    )
                    
                    current_volatility = self.calculate_portfolio_volatility(portfolio_history)

                    if signal == 1 and holdings[coin] == 0:  # Buy
                        position_size_usd = self.enhanced_position_sizing(strategy, cash, trade_log, current_volatility, coin)
                        amount = position_size_usd / current_price
                        exec_price = current_price * (1 + self.slippage_pct)
                        cost = amount * exec_price * (1 + self.transaction_cost)
                        
                        if cash >= cost:
                            holdings[coin] += amount
                            cash -= cost
                            entry_prices[coin] = exec_price
                            trade_log.append({'action': 'BUY', 'coin': coin, 'amount': amount, 'price': exec_price, 'value': cost, 'timestamp': datetime.now()})
                            self.performance_metrics['trades_executed'] += 1
                            
                            stop_loss_price = exec_price * (1 - strategy_params.get("stop_loss_pct", 0.05))
                            take_profit_price = exec_price * (1 + strategy_params.get("take_profit_pct", 0.10))
                            sl_order = Order(OrderType.STOP_LOSS, coin, stop_loss_price, amount, datetime.now(), oco_group=f"{coin}_SLTP")
                            tp_order = Order(OrderType.TAKE_PROFIT, coin, take_profit_price, amount, datetime.now(), oco_group=f"{coin}_SLTP")
                            open_orders[coin].extend([sl_order, tp_order])

                    elif signal == -1 and holdings[coin] > 0:  # Sell
                        amount_to_sell = holdings[coin]
                        exec_price = current_price * (1 - self.slippage_pct)
                        revenue = amount_to_sell * exec_price * (1 - self.transaction_cost)
                        cash += revenue
                        
                        profit = (exec_price - entry_prices[coin]) * amount_to_sell if entry_prices[coin] else 0
                        profit_pct = (exec_price - entry_prices[coin]) / entry_prices[coin] if entry_prices[coin] and entry_prices[coin] > 0 else 0
                        
                        trade_log.append({'action': 'SELL', 'coin': coin, 'amount': amount_to_sell, 'price': exec_price, 'value': revenue, 'profit': profit, 'profit_pct': profit_pct, 'timestamp': datetime.now()})
                        
                        holdings[coin] = 0
                        entry_prices[coin] = None
                        self.performance_metrics['trades_executed'] += 1
                        
                        open_orders[coin] = [o for o in open_orders[coin] if o.oco_group != f"{coin}_SLTP"]

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
                    self.enhanced_log('CRITICAL', f"Maximum drawdown ({self.max_drawdown_pct*100:.1f}%) reached: {current_drawdown*100:.1f}%", strategy)
                    self.circuit_breaker_active = True
                    self.circuit_breaker_trigger_time = datetime.now()
                    break 

                previous_prices = prices.copy()
                time.sleep(self.update_interval)

            except Exception as e:
                self.enhanced_log('ERROR', f"Error in enhanced trading loop: {e}", strategy)
                time.sleep(self.update_interval)

        # Final calculations
        final_value = portfolio_history[-1]['value'] if portfolio_history else self.starting_cash
        total_return = (final_value - self.starting_cash) / self.starting_cash
        risk_metrics = self.calculate_enhanced_risk_metrics(portfolio_history)

        results = {
            'Return': total_return * 100,
            'Final Value': final_value,
            'Trades': len([t for t in trade_log if t['action'] == 'BUY']),
            'Win Rate': self.calculate_win_rate(trade_log),
            'Sharpe Ratio': risk_metrics.get('sharpe_ratio', 0),
            'Sortino Ratio': risk_metrics.get('sortino_ratio', 0),
            'Max Drawdown (%)': risk_metrics.get('max_drawdown', 0) * 100,
            'Calmar Ratio': risk_metrics.get('calmar_ratio', 0),
            'Ulcer Index': risk_metrics.get('ulcer_index', 0),
            'Volatility': risk_metrics.get('volatility', 0)
        }
        
        self.enhanced_log('INFO', f"Strategy {strategy} finished. Final Value: ${final_value:.2f}", strategy)
        
        return {
            'results': results,
            'trade_log': trade_log,
            'portfolio_history': portfolio_history
        }

    def calculate_portfolio_volatility(self, portfolio_history: List) -> float:
        if len(portfolio_history) < 10: return 0.01
        try:
            values = [p.get('value', 0) for p in portfolio_history if 'value' in p]
            returns = pd.Series(values).pct_change().dropna()
            return returns.std() if not returns.empty else 0.01
        except Exception as e:
            self.enhanced_log('ERROR', f"Error calculating portfolio volatility: {e}")
            return 0.01

    def calculate_win_rate(self, trade_log: List) -> float:
        if not trade_log: return 0.0
        sells = [t for t in trade_log if t.get('profit_pct') is not None]
        if not sells: return 0.0
        profitable_trades = [t for t in sells if t.get('profit', 0) > 0]
        return len(profitable_trades) / len(sells) if len(sells) > 0 else 0.0

    def _calculate_signals_for_coin(self, coin: str, prices: Dict, strategy: str, lookback_period: int, 
                                    min_price_variation_pct: float, strategy_timeframes: List[str]) -> int:
        """Calculate trading signals for a specific coin (Mock Implementation)"""
        try:
            tf = strategy_timeframes[0]
            if tf not in self.historical_data_dfs or coin not in self.historical_data_dfs[tf]:
                return 0 # Not enough data
                
            history = self.historical_data_dfs[tf][coin].dropna()
            
            if len(history) < lookback_period:
                return 0 # Not enough data

            # ENHANCED SIGNAL CALCULATION FOR NEW STRATEGIES
            if strategy == "MeanReversion_Pro":
                # Enhanced mean reversion with volatility filter
                mean = history.mean()
                std = history.std()
                current_price = prices[coin]
                volatility = history.pct_change().std()
                
                if volatility > self.params[strategy].get("volatility_filter", 0.002):
                    if current_price < mean - 1.2 * std: return 1
                    elif current_price > mean + 1.2 * std: return -1
            
            elif strategy == "MA_Momentum_Hybrid":
                # MA + Momentum hybrid
                ma_short = history.iloc[-self.params[strategy].get("ma_short", 8):].mean()
                ma_long = history.iloc[-self.params[strategy].get("ma_long", 21):].mean()
                momentum = history.iloc[-1] / history.iloc[-self.params[strategy].get("momentum_period", 5)] - 1
                
                if (ma_short > ma_long * 1.002 and 
                    momentum > self.params[strategy].get("momentum_threshold", 0.02)):
                    return 1
                elif (ma_short < ma_long * 0.998 and 
                      momentum < -self.params[strategy].get("momentum_threshold", 0.02)):
                    return -1
            
            elif strategy == "Volatility_Regime_Adaptive":
                # New strategy: Adapt to volatility regimes
                short_vol = history.iloc[-self.params[strategy].get("low_vol_period", 10):].pct_change().std()
                long_vol = history.pct_change().std()
                
                if short_vol < self.params[strategy].get("vol_threshold", 0.005):
                    # Low volatility regime - use mean reversion
                    mean = history.mean()
                    current_price = prices[coin]
                    if current_price < mean * 0.99: return 1
                    elif current_price > mean * 1.01: return -1
                else:
                    # High volatility regime - use trend following
                    if history.iloc[-1] > history.iloc[-5:].mean(): return 1
                    elif history.iloc[-1] < history.iloc[-5:].mean(): return -1

            # ORIGINAL STRATEGIES
            elif "MA" in strategy:
                sma_short = history.iloc[-5:].mean()
                sma_long = history.iloc[-20:].mean()
                if sma_short > sma_long * 1.001: return 1
                elif sma_short < sma_long * 0.999: return -1
            
            elif "Momentum" in strategy:
                momentum = history.iloc[-1] / history.iloc[-5] - 1
                if momentum > 0.01: return 1
                elif momentum < -0.01: return -1

            elif "MeanReversion" in strategy:
                mean = history.mean()
                std = history.std()
                current_price = prices[coin]
                if current_price < mean - 1.5 * std: return 1
                elif current_price > mean + 1.5 * std: return -1

            return 0 
            
        except Exception as e:
            self.enhanced_log('ERROR', f"Error calculating signal for {coin}: {e}", strategy, coin)
            return 0

    def run_parallel_strategies(self, coins, strategies, duration_minutes=5, lookback_period=30):
        """Run multiple strategies in parallel using a thread pool"""
        self.start() # Start data fetching
        all_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            futures = {
                executor.submit(
                    self.run_enhanced_single_strategy, 
                    coins, 
                    strategy, 
                    duration_minutes, 
                    lookback_period
                ): strategy 
                for strategy in strategies
            }
            
            for future in concurrent.futures.as_completed(futures):
                strategy = futures[future]
                try:
                    result = future.result()
                    all_results[strategy] = result
                except Exception as e:
                    self.enhanced_log('ERROR', f"Strategy {strategy} failed: {e}", strategy)
                    all_results[strategy] = {'results': {}, 'trade_log': [], 'portfolio_history': []}
                    
        self.stop() # Stop data fetching
        self.enhanced_log('INFO', "Parallel simulation finished.")
        return all_results

    def real_time_monitoring_dashboard(self, all_results: Dict, current_prices: Dict):
        """Display a simple text-based real-time dashboard"""
        print("\n--- 📈 REAL-TIME MONITORING DASHBOARD (SNAPSHOT) ---")
        
        print("\n--- Current Prices (from CoinLore) ---")
        if not current_prices:
            print("No prices fetched. API might be down or rate-limited.")
        else:
            for i, (coin, price) in enumerate(current_prices.items()):
                print(f"{coin}: ${price:<10.4f}", end=" | ")
                if (i + 1) % 5 == 0:
                    print()
        
        print("\n\n--- Strategy Performance (Live) ---")
        print(f"{'Strategy':<25} | {'Value':<15} | {'Return %':<10} | {'Trades':<8} | {'Win Rate %':<10} | {'Max DD %':<10}")
        print("-" * 90)
        
        for strategy, data in all_results.items():
            res = data.get('results', {})
            print(f"{strategy:<25} | ${res.get('Final Value', 0):<14.2f} | {res.get('Return', 0):<9.2f}% | {res.get('Trades', 0):<8} | {res.get('Win Rate', 0)*100:<9.2f}% | {res.get('Max Drawdown (%)', 0):<9.2f}%")
        
        print("\n--- System Alerts ---")
        alerts_triggered = False
        for alert_type, is_active in self.alerts.items():
            if is_active:
                print(f"   [!] WARNING: {alert_type.replace('_', ' ').upper()} DETECTED")
                alerts_triggered = True
        
        if not alerts_triggered:
            print("   [✅] All systems nominal.")
        print("-" * 55)

    def export_results(self, all_results: Dict):
        """Export results to JSON and CSV"""
        try:
            summary = {s: d['results'] for s, d in all_results.items()}
            with open(os.path.join(self.output_dir, 'summary_results.json'), 'w') as f:
                json.dump(summary, f, indent=4)
            
            for strategy, data in all_results.items():
                if data['trade_log']:
                    df = pd.DataFrame(data['trade_log'])
                    df.to_csv(os.path.join(self.output_dir, f'trade_log_{strategy}.csv'), index=False)
                    
            self.enhanced_log('INFO', "Results exported successfully.")
            
        except Exception as e:
            self.enhanced_log('ERROR', f"Failed to export results: {e}")

    def generate_performance_report(self, all_results: Dict) -> Dict:
        """Generate comprehensive performance report"""
        report = {'summary': {}, 'strategy_comparison': {}, 'risk_analysis': {}, 'recommendations': []}
        try:
            for strategy, data in all_results.items():
                res = data.get('results', {})
                report['strategy_comparison'][strategy] = {
                    'return': res.get('Return', 0),
                    'sharpe': res.get('Sharpe Ratio', 0),
                    'max_drawdown': res.get('Max Drawdown (%)', 0),
                    'win_rate': res.get('Win Rate', 0),
                    'calmar_ratio': res.get('Calmar Ratio', 0)
                }

            if report['strategy_comparison']:
                best_strategy = max(report['strategy_comparison'].items(), 
                                    key=lambda x: x[1]['sharpe'], 
                                    default=(None, {}))
                
                report['summary'] = {
                    'best_strategy': best_strategy[0],
                    'best_sharpe': best_strategy[1].get('sharpe', 0),
                    'total_trades': sum(data.get('results', {}).get('Trades', 0) for data in all_results.values()),
                    'avg_win_rate': np.mean([data.get('results', {}).get('Win Rate', 0) for data in all_results.values()])
                }
                
        except Exception as e:
            self.enhanced_log('ERROR', f"Error generating performance report: {e}")
        return report

    def generate_advanced_charts(self, all_results: Dict, performance_report: Dict):
        """
        Génère tous les graphiques avancés en utilisant le AdvancedChartGenerator.
        """
        chart_files = {}
        try:
            chart_files['comprehensive_dashboard'] = self.chart_generator.create_comprehensive_dashboard(all_results, performance_report)
            chart_files['interactive_dashboard'] = self.chart_generator.create_interactive_dashboard(all_results)
            chart_files['animated_performance'] = self.chart_generator.create_animated_performance_chart(all_results)
            chart_files['cumulative_returns'] = self.chart_generator.plot_cumulative_returns(all_results)
            return chart_files
        except Exception as e:
            self.enhanced_log('ERROR', f"Erreur lors de la génération des graphiques: {e}")
            return {}

if __name__ == "__main__":
    simulator = EnhancedParallelCryptoSimulatorCG(
        starting_cash=10, #sim 1 2: 5000,
        update_interval=10, #10 #3
        max_drawdown_pct=0.20,
        min_cash_pct=0.10,
        max_position_concentration=0.25
    )

    all_coins = list(simulator.coinlore_id_map.keys())
    
    # UPDATED: Only run the top 3 + 2 enhanced + 1 new strategy
    selected_strategies = [
        "MeanReversion",           # Top 1
        "MA_Enhanced",             # Top 2  
        "Momentum_Enhanced",       # Top 3
        "MeanReversion_Pro",       # Enhanced version
        "MA_Momentum_Hybrid",      # Hybrid strategy
        "Volatility_Regime_Adaptive" # Completely new
    ]

    print("Starting enhanced cryptocurrency trading simulation with REAL CoinLore data...")
    print(f"Running optimized strategy set: {selected_strategies}")
    
    simulation_results = simulator.run_parallel_strategies(
        coins=all_coins,
        strategies=selected_strategies,
        duration_minutes=300,
        lookback_period=30 #30
    )

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
    
    simulator.real_time_monitoring_dashboard(simulation_results, current_prices)
