"""
Real-time Risk Dashboard
Live monitoring of portfolio risk and performance
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import threading
import asyncio
from collections import deque

logger = logging.getLogger(__name__)

class RiskDashboard:
    """Interactive risk monitoring dashboard"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.app = dash.Dash(__name__)
        
        # Data storage
        self.portfolio_data = deque(maxlen=1000)
        self.position_data = deque(maxlen=500)
        self.risk_metrics = deque(maxlen=100)
        self.alerts = deque(maxlen=100)
        
        # Initialize dashboard
        self._setup_layout()
        self._setup_callbacks()
        
        # Background update thread
        self.update_thread = None
        self.running = False
        
        logger.info("Risk dashboard initialized")
    
    def _setup_layout(self):
        """Setup dashboard layout"""
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Institutional Trading System - Risk Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.Div(id='system-status', 
                        style={'textAlign': 'center', 'marginBottom': '20px'})
            ], className='row'),
            
            # First Row: Key Metrics
            html.Div([
                # Portfolio Metrics
                html.Div([
                    html.H3("Portfolio Metrics", style={'color': '#3498db'}),
                    html.Div(id='portfolio-metrics', 
                            style={'fontSize': '16px', 'lineHeight': '1.8'})
                ], className='four columns', 
                   style={'backgroundColor': '#f8f9fa', 'padding': '15px', 
                          'borderRadius': '5px', 'margin': '5px'}),
                
                # Risk Metrics
                html.Div([
                    html.H3("Risk Metrics", style={'color': '#e74c3c'}),
                    html.Div(id='risk-metrics',
                            style={'fontSize': '16px', 'lineHeight': '1.8'})
                ], className='four columns',
                   style={'backgroundColor': '#f8f9fa', 'padding': '15px',
                          'borderRadius': '5px', 'margin': '5px'}),
                
                # Active Positions
                html.Div([
                    html.H3("Active Positions", style={'color': '#27ae60'}),
                    html.Div(id='active-positions',
                            style={'fontSize': '14px', 'lineHeight': '1.6'})
                ], className='four columns',
                   style={'backgroundColor': '#f8f9fa', 'padding': '15px',
                          'borderRadius': '5px', 'margin': '5px'})
            ], className='row'),
            
            # Second Row: Charts
            html.Div([
                # Equity Curve
                html.Div([
                    html.H4("Portfolio Equity Curve"),
                    dcc.Graph(id='equity-curve'),
                    dcc.Interval(id='equity-update', interval=5000)
                ], className='six columns',
                   style={'backgroundColor': 'white', 'padding': '10px',
                          'borderRadius': '5px', 'margin': '5px',
                          'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                
                # Risk Heatmap
                html.Div([
                    html.H4("Risk Exposure Heatmap"),
                    dcc.Graph(id='risk-heatmap'),
                    dcc.Interval(id='heatmap-update', interval=10000)
                ], className='six columns',
                   style={'backgroundColor': 'white', 'padding': '10px',
                          'borderRadius': '5px', 'margin': '5px',
                          'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], className='row'),
            
            # Third Row: Alerts and Drawdown
            html.Div([
                # Recent Alerts
                html.Div([
                    html.H4("Recent Alerts", style={'color': '#e74c3c'}),
                    html.Div(id='alert-list',
                            style={'maxHeight': '300px', 'overflowY': 'auto',
                                   'fontSize': '13px'})
                ], className='six columns',
                   style={'backgroundColor': '#fff5f5', 'padding': '15px',
                          'borderRadius': '5px', 'margin': '5px',
                          'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                
                # Drawdown Analysis
                html.Div([
                    html.H4("Drawdown Analysis"),
                    dcc.Graph(id='drawdown-chart'),
                    dcc.Interval(id='drawdown-update', interval=15000)
                ], className='six columns',
                   style={'backgroundColor': 'white', 'padding': '10px',
                          'borderRadius': '5px', 'margin': '5px',
                          'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], className='row'),
            
            # Fourth Row: Performance Attribution
            html.Div([
                html.Div([
                    html.H4("Performance Attribution"),
                    dcc.Graph(id='attribution-chart'),
                    dcc.Interval(id='attribution-update', interval=20000)
                ], className='twelve columns',
                   style={'backgroundColor': 'white', 'padding': '10px',
                          'borderRadius': '5px', 'margin': '5px',
                          'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], className='row'),
            
            # Control Panel
            html.Div([
                html.H4("Dashboard Controls"),
                html.Button('Refresh All', id='refresh-btn', n_clicks=0,
                          style={'margin': '5px', 'padding': '10px 20px'}),
                html.Button('Clear Alerts', id='clear-alerts-btn', n_clicks=0,
                          style={'margin': '5px', 'padding': '10px 20px'}),
                dcc.Interval(id='control-interval', interval=1000)
            ], className='row', 
               style={'textAlign': 'center', 'marginTop': '20px'})
        ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'})
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            Output('system-status', 'children'),
            Input('control-interval', 'n_intervals')
        )
        def update_system_status(n):
            return self._get_system_status()
        
        @self.app.callback(
            Output('portfolio-metrics', 'children'),
            Input('control-interval', 'n_intervals')
        )
        def update_portfolio_metrics(n):
            return self._get_portfolio_metrics()
        
        @self.app.callback(
            Output('risk-metrics', 'children'),
            Input('control-interval', 'n_intervals')
        )
        def update_risk_metrics(n):
            return self._get_risk_metrics()
        
        @self.app.callback(
            Output('active-positions', 'children'),
            Input('control-interval', 'n_intervals')
        )
        def update_active_positions(n):
            return self._get_active_positions()
        
        @self.app.callback(
            Output('equity-curve', 'figure'),
            Input('equity-update', 'n_intervals')
        )
        def update_equity_curve(n):
            return self._create_equity_chart()
        
        @self.app.callback(
            Output('risk-heatmap', 'figure'),
            Input('heatmap-update', 'n_intervals')
        )
        def update_risk_heatmap(n):
            return self._create_risk_heatmap()
        
        @self.app.callback(
            Output('alert-list', 'children'),
            Input('control-interval', 'n_intervals')
        )
        def update_alerts(n):
            return self._get_alerts_list()
        
        @self.app.callback(
            Output('drawdown-chart', 'figure'),
            Input('drawdown-update', 'n_intervals')
        )
        def update_drawdown_chart(n):
            return self._create_drawdown_chart()
        
        @self.app.callback(
            Output('attribution-chart', 'figure'),
            Input('attribution-update', 'n_intervals')
        )
        def update_attribution_chart(n):
            return self._create_attribution_chart()
        
        @self.app.callback(
            Output('alert-list', 'children', allow_duplicate=True),
            Input('clear-alerts-btn', 'n_clicks'),
            prevent_initial_call=True
        )
        def clear_alerts(n_clicks):
            self.alerts.clear()
            return self._get_alerts_list()
    
    def _get_system_status(self) -> str:
        """Get current system status"""
        
        status_colors = {
            'running': 'green',
            'warning': 'orange',
            'error': 'red',
            'stopped': 'gray'
        }
        
        status = "🟢 System Running"  # Default
        
        # Check for recent alerts
        if self.alerts:
            recent_critical = any(
                alert.get('level') == 'CRITICAL' 
                for alert in list(self.alerts)[-5:]
            )
            
            if recent_critical:
                status = "🔴 System Error - Critical Alerts"
            else:
                status = "🟡 System Warning - Check Alerts"
        
        return html.Div([
            html.Span(status, style={'fontSize': '18px', 'fontWeight': 'bold'}),
            html.Br(),
            html.Span(f"Last Update: {datetime.now().strftime('%H:%M:%S')}",
                     style={'fontSize': '14px', 'color': '#7f8c8d'})
        ])
    
    def _get_portfolio_metrics(self) -> List:
        """Get portfolio metrics display"""
        
        if not self.portfolio_data:
            return ["No portfolio data available"]
        
        latest = list(self.portfolio_data)[-1] if self.portfolio_data else {}
        
        metrics = [
            html.Div([
                html.Strong("Portfolio Value: "),
                html.Span(f"${latest.get('value', 0):,.2f}", 
                         style={'color': '#27ae60' if latest.get('value', 0) >= 0 else '#e74c3c'})
            ]),
            html.Div([
                html.Strong("Daily P&L: "),
                html.Span(f"${latest.get('daily_pnl', 0):+,.2f}",
                         style={'color': '#27ae60' if latest.get('daily_pnl', 0) >= 0 else '#e74c3c'})
            ]),
            html.Div([
                html.Strong("Daily Return: "),
                html.Span(f"{latest.get('daily_return', 0):+.2%}",
                         style={'color': '#27ae60' if latest.get('daily_return', 0) >= 0 else '#e74c3c'})
            ]),
            html.Div([
                html.Strong("Total P&L: "),
                html.Span(f"${latest.get('total_pnl', 0):+,.2f}",
                         style={'color': '#27ae60' if latest.get('total_pnl', 0) >= 0 else '#e74c3c'})
            ]),
            html.Div([
                html.Strong("Cash Balance: "),
                html.Span(f"${latest.get('cash', 0):,.2f}")
            ]),
            html.Div([
                html.Strong("Leverage: "),
                html.Span(f"{latest.get('leverage', 0):.2f}x",
                         style={'color': '#e74c3c' if latest.get('leverage', 0) > 2 else '#3498db'})
            ])
        ]
        
        return metrics
    
    def _get_risk_metrics(self) -> List:
        """Get risk metrics display"""
        
        if not self.risk_metrics:
            return ["No risk data available"]
        
        latest = list(self.risk_metrics)[-1] if self.risk_metrics else {}
        
        # Calculate drawdown status
        drawdown = latest.get('max_drawdown', 0)
        if drawdown > 0.1:  # 10%
            drawdown_color = '#e74c3c'
        elif drawdown > 0.05:  # 5%
            drawdown_color = '#f39c12'
        else:
            drawdown_color = '#27ae60'
        
        metrics = [
            html.Div([
                html.Strong("Max Drawdown: "),
                html.Span(f"{drawdown:.2%}", style={'color': drawdown_color})
            ]),
            html.Div([
                html.Strong("Sharpe Ratio: "),
                html.Span(f"{latest.get('sharpe_ratio', 0):.2f}",
                         style={'color': '#27ae60' if latest.get('sharpe_ratio', 0) > 1 else '#e74c3c'})
            ]),
            html.Div([
                html.Strong("Sortino Ratio: "),
                html.Span(f"{latest.get('sortino_ratio', 0):.2f}",
                         style={'color': '#27ae60' if latest.get('sortino_ratio', 0) > 1 else '#e74c3c'})
            ]),
            html.Div([
                html.Strong("VaR (95%): "),
                html.Span(f"{latest.get('var_95', 0):.2%}")
            ]),
            html.Div([
                html.Strong("Expected Shortfall: "),
                html.Span(f"{latest.get('expected_shortfall', 0):.2%}")
            ]),
            html.Div([
                html.Strong("Portfolio Beta: "),
                html.Span(f"{latest.get('portfolio_beta', 0):.2f}")
            ])
        ]
        
        return metrics
    
    def _get_active_positions(self) -> List:
        """Get active positions display"""
        
        if not self.position_data:
            return ["No active positions"]
        
        positions = list(self.position_data)
        
        position_list = []
        for pos in positions[-10:]:  # Last 10 positions
            pnl = pos.get('pnl', 0)
            pnl_pct = pos.get('pnl_pct', 0)
            
            position_list.append(html.Div([
                html.Strong(f"{pos.get('symbol', 'N/A')}: "),
                html.Span(f"{pos.get('side', 'N/A')} {pos.get('size', 0):.2f} @ ${pos.get('entry_price', 0):.2f}",
                         style={'fontSize': '12px'}),
                html.Br(),
                html.Span(f"P&L: ${pnl:+,.2f} ({pnl_pct:+.2%})",
                         style={'fontSize': '11px', 
                                'color': '#27ae60' if pnl >= 0 else '#e74c3c'})
            ], style={'marginBottom': '5px', 'padding': '3px', 
                     'backgroundColor': '#f8f9fa', 'borderRadius': '3px'}))
        
        return position_list
    
    def _create_equity_chart(self) -> go.Figure:
        """Create equity curve chart"""
        
        if not self.portfolio_data:
            return self._empty_chart("No portfolio data")
        
        # Prepare data
        portfolio_list = list(self.portfolio_data)
        timestamps = [p.get('timestamp') for p in portfolio_list]
        values = [p.get('value', 0) for p in portfolio_list]
        daily_pnl = [p.get('daily_pnl', 0) for p in portfolio_list]
        
        # Create figure with secondary y-axis
        fig = sp.make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value', 'Daily P&L'),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#3498db', width=2),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.1)'
            ),
            row=1, col=1
        )
        
        # Daily P&L
        colors = ['#27ae60' if pnl >= 0 else '#e74c3c' for pnl in daily_pnl]
        
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=daily_pnl,
                name='Daily P&L',
                marker_color=colors,
                marker_line_width=0
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=30, b=50)
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
        
        return fig
    
    def _create_risk_heatmap(self) -> go.Figure:
        """Create risk exposure heatmap"""
        
        if not self.position_data:
            return self._empty_chart("No position data")
        
        positions = list(self.position_data)
        
        # Group by symbol and calculate exposure
        exposure_data = {}
        for pos in positions:
            symbol = pos.get('symbol')
            exposure = pos.get('size', 0) * pos.get('entry_price', 0)
            
            if symbol not in exposure_data:
                exposure_data[symbol] = {
                    'exposure': 0,
                    'pnl': 0,
                    'count': 0
                }
            
            exposure_data[symbol]['exposure'] += exposure
            exposure_data[symbol]['pnl'] += pos.get('pnl', 0)
            exposure_data[symbol]['count'] += 1
        
        if not exposure_data:
            return self._empty_chart("No exposure data")
        
        # Prepare heatmap data
        symbols = list(exposure_data.keys())
        exposures = [data['exposure'] for data in exposure_data.values()]
        pnls = [data['pnl'] for data in exposure_data.values()]
        
        # Normalize for heatmap
        max_exposure = max(exposures) if exposures else 1
        normalized_exposures = [e / max_exposure for e in exposures]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[normalized_exposures],
            x=symbols,
            y=['Exposure'],
            colorscale='RdYlGn',
            showscale=True,
            hovertemplate='Symbol: %{x}<br>Exposure: $%{customdata[0]:,.0f}<br>P&L: $%{customdata[1]:+,.0f}<extra></extra>',
            customdata=np.array([[exposures, pnls]]).T
        ))
        
        # Add annotations for P&L
        for i, symbol in enumerate(symbols):
            fig.add_annotation(
                x=symbol,
                y=0,
                text=f"${pnls[i]:+,.0f}",
                showarrow=False,
                font=dict(
                    size=10,
                    color='white' if abs(normalized_exposures[i]) > 0.5 else 'black'
                )
            )
        
        fig.update_layout(
            height=300,
            title="Risk Exposure by Symbol",
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    def _get_alerts_list(self) -> List:
        """Get alerts list display"""
        
        if not self.alerts:
            return ["No recent alerts"]
        
        alerts_list = []
        for alert in list(self.alerts)[-20:]:  # Last 20 alerts
            level = alert.get('level', 'INFO')
            message = alert.get('message', '')
            timestamp = alert.get('timestamp', datetime.now())
            
            # Color coding
            if level == 'CRITICAL':
                color = '#e74c3c'
                icon = '🔴'
            elif level == 'ERROR':
                color = '#e67e22'
                icon = '🟠'
            elif level == 'WARNING':
                color = '#f1c40f'
                icon = '🟡'
            else:
                color = '#3498db'
                icon = '🔵'
            
            alerts_list.append(html.Div([
                html.Span(f"{icon} ", style={'marginRight': '5px'}),
                html.Strong(f"[{level}] ", style={'color': color}),
                html.Span(message, style={'fontSize': '12px'}),
                html.Br(),
                html.Span(timestamp.strftime('%H:%M:%S'), 
                         style={'fontSize': '10px', 'color': '#7f8c8d'})
            ], style={'marginBottom': '8px', 'padding': '8px',
                     'backgroundColor': '#f8f9fa', 'borderRadius': '4px',
                     'borderLeft': f'4px solid {color}'}))
        
        return alerts_list
    
    def _create_drawdown_chart(self) -> go.Figure:
        """Create drawdown analysis chart"""
        
        if not self.portfolio_data:
            return self._empty_chart("No portfolio data")
        
        portfolio_list = list(self.portfolio_data)
        values = [p.get('value', 0) for p in portfolio_list]
        timestamps = [p.get('timestamp') for p in portfolio_list]
        
        # Calculate drawdown
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max * 100
        
        fig = go.Figure()
        
        # Drawdown area
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=drawdowns,
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.3)',
            line=dict(color='#e74c3c', width=2),
            name='Drawdown',
            hovertemplate='Drawdown: %{y:.2f}%<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Current drawdown
        current_dd = drawdowns[-1] if drawdowns.size > 0 else 0
        fig.add_hline(y=current_dd, line_dash="dot", 
                     line_color="#e74c3c", opacity=0.7,
                     annotation_text=f"Current: {current_dd:.2f}%")
        
        fig.update_layout(
            height=300,
            title="Portfolio Drawdown",
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=50),
            yaxis=dict(title="Drawdown (%)", ticksuffix="%"),
            xaxis=dict(title="Time")
        )
        
        return fig
    
    def _create_attribution_chart(self) -> go.Figure:
        """Create performance attribution chart"""
        
        if not self.position_data:
            return self._empty_chart("No position data")
        
        positions = list(self.position_data)
        
        # Group P&L by symbol
        pnl_by_symbol = {}
        for pos in positions:
            symbol = pos.get('symbol')
            pnl = pos.get('pnl', 0)
            
            if symbol not in pnl_by_symbol:
                pnl_by_symbol[symbol] = 0
            
            pnl_by_symbol[symbol] += pnl
        
        if not pnl_by_symbol:
            return self._empty_chart("No P&L data")
        
        # Prepare data
        symbols = list(pnl_by_symbol.keys())
        pnls = list(pnl_by_symbol.values())
        
        # Sort by absolute P&L
        sorted_data = sorted(zip(symbols, pnls), key=lambda x: abs(x[1]), reverse=True)
        symbols = [s for s, _ in sorted_data][:15]  # Top 15
        pnls = [p for _, p in sorted_data][:15]
        
        # Colors
        colors = ['#27ae60' if p >= 0 else '#e74c3c' for p in pnls]
        
        fig = go.Figure(data=[
            go.Bar(
                x=symbols,
                y=pnls,
                marker_color=colors,
                text=[f"${p:+,.0f}" for p in pnls],
                textposition='auto',
                hovertemplate='Symbol: %{x}<br>P&L: $%{y:+,.0f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            height=350,
            title="Performance Attribution by Symbol",
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=100),
            xaxis=dict(tickangle=45, title="Symbol"),
            yaxis=dict(title="P&L ($)")
        )
        
        return fig
    
    def _empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def add_portfolio_data(self, data: Dict):
        """Add portfolio data point"""
        data['timestamp'] = datetime.now()
        self.portfolio_data.append(data)
    
    def add_position_data(self, data: Dict):
        """Add position data"""
        self.position_data.append(data)
    
    def add_risk_metrics(self, metrics: Dict):
        """Add risk metrics"""
        self.risk_metrics.append(metrics)
    
    def add_alert(self, level: str, message: str, data: Dict = None):
        """Add alert to dashboard"""
        
        alert = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message,
            'data': data or {}
        }
        
        self.alerts.append(alert)
        
        # Log based on level
        if level == 'CRITICAL':
            logger.critical(f"ALERT: {message}")
        elif level == 'ERROR':
            logger.error(f"ALERT: {message}")
        elif level == 'WARNING':
            logger.warning(f"ALERT: {message}")
        else:
            logger.info(f"ALERT: {message}")
    
    def start(self, port: int = 8050, debug: bool = False):
        """Start dashboard server"""
        
        def run_server():
            self.app.run_server(debug=debug, port=port, use_reloader=False)
        
        self.running = True
        self.update_thread = threading.Thread(target=run_server, daemon=True)
        self.update_thread.start()
        
        logger.info(f"Risk dashboard started on http://localhost:{port}")
    
    def shutdown(self):
        """Shutdown dashboard"""
        self.running = False
        logger.info("Risk dashboard shutdown")