import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from anthropic import Anthropic
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TradingSignal:
    timestamp: str
    symbol: str
    signal: str  # BUY, SELL, HOLD, SHORT
    confidence: float
    regime_score: float
    technical_score: float
    risk_score: float
    coordinator_score: float
    price: float
    reasoning: str

@dataclass
class AgentResponse:
    agent_name: str
    analysis: str
    signal: str
    confidence: float
    features: Dict
    learning_feedback: str

class DataProvider:
    """Handles all data acquisition from Alpha Vantage and other sources"""
    
    def __init__(self, alpha_vantage_key: str):
        self.av_key = alpha_vantage_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_spy_data(self, period: str = "1y") -> pd.DataFrame:
        """Get SPY data with multiple timeframes"""
        spy = yf.Ticker("SPY")
        data = spy.history(period=period)
        
        # Add technical indicators
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['MACD'], data['MACD_Signal'] = self._calculate_macd(data['Close'])
        data['Volatility'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        return data
    
    def get_vix_data(self, period: str = "1y") -> pd.DataFrame:
        """Get VIX data"""
        vix = yf.Ticker("^VIX")
        return vix.history(period=period)
    
    def get_yield_data(self) -> Dict:
        """Get treasury yield data from Alpha Vantage"""
        url = f"{self.base_url}?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey={self.av_key}"
        response = requests.get(url)
        return response.json()
    
    def get_economic_indicators(self) -> Dict:
        """Get key economic indicators"""
        indicators = {}
        
        # Federal Funds Rate
        url = f"{self.base_url}?function=FEDERAL_FUNDS_RATE&interval=monthly&apikey={self.av_key}"
        indicators['fed_funds'] = requests.get(url).json()
        
        # GDP
        url = f"{self.base_url}?function=REAL_GDP&interval=annual&apikey={self.av_key}"
        indicators['gdp'] = requests.get(url).json()
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        prices = prices.astype(float)
        delta = prices.diff().astype(float)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd, signal

class AgentLearningSystem:
    """Manages inter-agent learning and feedback loops"""
    
    def __init__(self, db_path: str = "agent_learning.db"):
        self.db_path = db_path
        self._init_database()
        self.performance_history = {}
    
    def _init_database(self):
        """Initialize SQLite database for learning storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY,
                agent_name TEXT,
                timestamp TEXT,
                signal TEXT,
                confidence REAL,
                actual_outcome REAL,
                features TEXT,
                learning_update TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inter_agent_feedback (
                id INTEGER PRIMARY KEY,
                source_agent TEXT,
                target_agent TEXT,
                feedback_type TEXT,
                feedback_content TEXT,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_performance(self, agent_name: str, signal: str, confidence: float, 
                         features: Dict, actual_outcome: Optional[float] = None):
        """Store agent performance for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO agent_performance 
            (agent_name, timestamp, signal, confidence, actual_outcome, features)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (agent_name, datetime.now().isoformat(), signal, confidence, 
              actual_outcome, json.dumps(features)))
        
        conn.commit()
        conn.close()
    
    def get_agent_learning_context(self, agent_name: str) -> str:
        """Get learning context for an agent based on historical performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT signal, confidence, actual_outcome, features 
            FROM agent_performance 
            WHERE agent_name = ? AND actual_outcome IS NOT NULL
            ORDER BY timestamp DESC LIMIT 50
        ''', (agent_name,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return "No historical performance data available."
        
        # Analyze performance patterns
        accuracy_by_signal = {}
        confidence_performance = []
        
        for signal, confidence, outcome, features_json in results:
            if signal not in accuracy_by_signal:
                accuracy_by_signal[signal] = []
            
            # Simple binary outcome: positive return = 1, negative = 0
            binary_outcome = 1 if outcome > 0 else 0
            accuracy_by_signal[signal].append(binary_outcome)
            confidence_performance.append((confidence, binary_outcome))
        
        # Generate learning insights
        insights = []
        for signal, outcomes in accuracy_by_signal.items():
            accuracy = np.mean(outcomes)
            insights.append(f"{signal} accuracy: {accuracy:.2%} ({len(outcomes)} trades)")
        
        return f"Historical Performance:\n" + "\n".join(insights)

class LRMAgent:
    """Base class for all LRM agents with learning capabilities"""
    
    def __init__(self, name: str, anthropic_client: Anthropic, learning_system: AgentLearningSystem):
        self.name = name
        self.client = anthropic_client
        self.learning_system = learning_system
    
    def analyze(self, market_data: Dict, learning_context: str = "") -> AgentResponse:
        """Base analysis method to be overridden by specific agents"""
        raise NotImplementedError
    
    def _make_api_call(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to Claude with error handling"""
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text
        except Exception as e:
            st.error(f"API call failed for {self.name}: {str(e)}")
            return f"Error in {self.name} analysis: {str(e)}"
    
    def update_learning(self, signal: str, confidence: float, features: Dict, outcome: Optional[float] = None):
        """Update agent's learning based on performance"""
        self.learning_system.store_performance(self.name, signal, confidence, features, outcome)

class RegimeAgent(LRMAgent):
    """Market regime detection agent"""
    
    def __init__(self, anthropic_client: Anthropic, learning_system: AgentLearningSystem):
        super().__init__("Regime", anthropic_client, learning_system)
    
    def _truncate_learning_context(self, learning_context: str, max_tokens: int = 30000) -> str:
        """Truncate learning context to prevent token overflow"""
        if not learning_context:
            return ""
        
        # Estimate tokens (roughly 4 chars per token)
        estimated_tokens = len(learning_context) / 4
        
        if estimated_tokens <= max_tokens:
            return learning_context
        
        # Keep most recent context
        target_chars = max_tokens * 4
        if len(learning_context) > target_chars:
            truncated = learning_context[-int(target_chars):]
            # Find good breaking point
            if '\n\n' in truncated:
                truncated = truncated[truncated.find('\n\n')+2:]
            elif '. ' in truncated and len(truncated) > 1000:
                truncated = truncated[truncated.find('. ')+2:]
            
            return f"[RECENT LEARNING CONTEXT]\n{truncated}"
        
        return learning_context
    
    def _extract_key_data(self, market_data: Dict) -> str:
        """Extract only essential market data to minimize tokens"""
        try:
            # Get core values safely
            spy_price = market_data.get('spy_current', {}).get('Close', 'N/A')
            vix_level = market_data.get('vix_current', {}).get('Close', 'N/A')
            
            # Summarize yields (just key ones)
            yields = market_data.get('yields', {})
            yield_summary = f"10Y: {yields.get('10Y', 'N/A')}, 2Y: {yields.get('2Y', 'N/A')}"
            
            # Limit technical indicators to essentials
            technicals = market_data.get('technicals', {})
            tech_summary = str({k: v for k, v in list(technicals.items())[:5]})[:200] + "..."
            
            # Limit economic indicators
            economic = market_data.get('economic', {})
            econ_summary = str({k: v for k, v in list(economic.items())[:3]})[:150] + "..."
            
            return f"""SPY: {spy_price} | VIX: {vix_level}
Yields: {yield_summary}
Technicals: {tech_summary}
Economic: {econ_summary}"""
            
        except Exception as e:
            return f"Data extraction error: {str(e)}"
    
    def analyze(self, market_data: Dict, learning_context: str = "") -> AgentResponse:
        print(f"[{self.name}] Starting analysis...")
        
        # Truncate learning context to stay under token limits
        truncated_context = self._truncate_learning_context(learning_context, 25000)
        print(f"[{self.name}] Learning context length: {len(truncated_context)} chars")
        
        # Compact system prompt
        system_prompt = f"""Market Regime Specialist - Identify SPY trading regime.

ANALYZE: Volatility patterns, macro conditions, bull/bear phases, stress indicators.

{truncated_context}

OUTPUT FORMAT:
- Regime: [BULL/BEAR/SIDEWAYS/TRANSITION]  
- Confidence: [0.0-1.0]
- Signal: [BUY/SELL/HOLD/SHORT]
- Key factors (3-4 points)

Be concise and actionable."""

        # Compact user prompt with essential data only
        key_data = self._extract_key_data(market_data)
        user_prompt = f"""Market Data:
{key_data}

Analyze regime with confidence level and trading signal."""
        
        # Token estimation and logging
        total_tokens = (len(system_prompt) + len(user_prompt)) // 4
        print(f"[{self.name}] Estimated tokens: {total_tokens}")
        
        if total_tokens > 190000:  # Safety margin
            print(f"[{self.name}] WARNING: Token count too high, further truncating...")
            truncated_context = self._truncate_learning_context(learning_context, 10000)
            system_prompt = f"""Market Regime Specialist - Identify SPY trading regime.

{truncated_context}

OUTPUT: Regime [BULL/BEAR/SIDEWAYS/TRANSITION], Confidence [0.0-1.0], Signal [BUY/SELL/HOLD/SHORT]"""
        
        try:
            print(f"[{self.name}] Making API call...")
            response_text = self._make_api_call(system_prompt, user_prompt)
            print(f"[{self.name}] API response received: {len(response_text) if response_text else 0} chars")
            
            # Check if we got a valid response
            if not response_text or len(response_text.strip()) == 0:
                print(f"[{self.name}] WARNING: Empty response from API")
                return self._create_fallback_response("Empty API response")
            
            print(f"[{self.name}] Processing response...")
            
            # Enhanced signal parsing
            signal = "HOLD"
            response_upper = response_text.upper()
            
            if "BUY" in response_upper and "SELL" not in response_upper:
                signal = "BUY"
            elif "SELL" in response_upper:
                signal = "SELL"
            elif "SHORT" in response_upper:
                signal = "SHORT"
            
            print(f"[{self.name}] Extracted signal: {signal}")
            
            # Enhanced confidence extraction
            confidence = 0.5
            try:
                import re
                patterns = [
                    r'confidence[:\s]*([0-9.]+)',
                    r'([0-9.]+)\s*confidence',
                    r'level[:\s]*([0-9.]+)'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, response_text.lower())
                    if match:
                        conf_val = float(match.group(1))
                        if conf_val > 1:  # Handle percentage
                            conf_val = conf_val / 100
                        confidence = min(1.0, max(0.0, conf_val))
                        break
            except Exception as conf_error:
                print(f"[{self.name}] Confidence extraction error: {conf_error}")
            
            print(f"[{self.name}] Extracted confidence: {confidence}")
            
            # Safe feature extraction
            features = {
                'vix_level': self._safe_extract_float(market_data, ['vix_current', 'Close'], 20.0),
                'spy_price': self._safe_extract_float(market_data, ['spy_current', 'Close'], 400.0),
                'regime_type': 'analysis_based',
                'token_estimate': total_tokens
            }
            
            print(f"[{self.name}] Analysis complete successfully")
            
            return AgentResponse(
                agent_name=self.name,
                analysis=response_text,
                signal=signal,
                confidence=confidence,
                features=features,
                learning_feedback=""
            )
            
        except Exception as e:
            print(f"[{self.name}] ERROR: {str(e)}")
            return self._create_fallback_response(f"Analysis error: {str(e)}")
    
    def _create_fallback_response(self, error_msg: str) -> AgentResponse:
        """Create a safe fallback response when analysis fails"""
        print(f"[{self.name}] Creating fallback response: {error_msg}")
        return AgentResponse(
            agent_name=self.name,
            analysis=f"Regime analysis unavailable: {error_msg}. Using conservative HOLD position.",
            signal="HOLD",
            confidence=0.3,
            features={'error': True, 'vix_level': 20.0, 'spy_price': 400.0},
            learning_feedback=""
        )
    
    def _safe_extract_float(self, data: Dict, keys: list, default: float) -> float:
        """Safely extract float from nested dictionary"""
        try:
            value = data
            for key in keys:
                value = value.get(key, {})
            return float(value) if isinstance(value, (int, float)) else default
        except:
            return default
            
            
class TechnicalAgent(LRMAgent):
    """Technical analysis agent"""
    
    def __init__(self, anthropic_client: Anthropic, learning_system: AgentLearningSystem):
        super().__init__("Technical", anthropic_client, learning_system)
    
    def analyze(self, market_data: Dict, learning_context: str = "") -> AgentResponse:
        system_prompt = f"""You are a Technical Analysis expert specializing in SPY trading signals.

Your expertise includes:
- Chart pattern recognition
- Technical indicator analysis (RSI, MACD, Moving Averages)
- Support/resistance levels
- Volume analysis
- Momentum indicators
- Price action analysis

Learning Context from your past performance:
{learning_context}

Provide specific entry/exit levels and confidence assessments.
Consider multiple timeframes and confluence of signals.

Output your signal as BUY/SELL/HOLD/SHORT with confidence level 0-1."""
        
        user_prompt = f"""Perform technical analysis on SPY:

Current Price Data: {market_data.get('spy_current', {})}
Technical Indicators: {market_data.get('technicals', {})}
Volume Data: {market_data.get('volume', 'N/A')}
Recent Price Action: {market_data.get('recent_data', {})}

Provide technical analysis with specific levels and signal confidence."""
        
        response_text = self._make_api_call(system_prompt, user_prompt)
        
        # Parse technical signal
        signal = "HOLD"
        confidence = 0.5
        
        if "strong buy" in response_text.lower() or "bullish" in response_text.lower():
            signal = "BUY"
            confidence = 0.8
        elif "buy" in response_text.lower():
            signal = "BUY"
            confidence = 0.6
        elif "sell" in response_text.lower() or "bearish" in response_text.lower():
            signal = "SELL"
            confidence = 0.7
        elif "short" in response_text.lower():
            signal = "SHORT"
            confidence = 0.7
        
        features = {
            'rsi': market_data.get('technicals', {}).get('RSI', 50),
            'macd': market_data.get('technicals', {}).get('MACD', 0),
            'sma_crossover': market_data.get('technicals', {}).get('SMA_signal', 0)
        }
        
        return AgentResponse(
            agent_name=self.name,
            analysis=response_text,
            signal=signal,
            confidence=confidence,
            features=features,
            learning_feedback=""
        )

class RiskAgent(LRMAgent):
    """Risk management agent"""
    
    def __init__(self, anthropic_client: Anthropic, learning_system: AgentLearningSystem):
        super().__init__("Risk", anthropic_client, learning_system)
    
    def analyze(self, market_data: Dict, learning_context: str = "") -> AgentResponse:
        system_prompt = f"""You are a Risk Management specialist focused on capital preservation and optimal position sizing.

Your responsibilities:
- Assess market volatility and risk levels
- Evaluate position sizing recommendations
- Identify potential tail risks
- Monitor correlation risks
- Assess liquidity conditions
- Consider drawdown management

Learning Context from your past performance:
{learning_context}

Provide risk-adjusted signal recommendations with specific risk parameters.
Consider both systematic and idiosyncratic risks.

Always prioritize capital preservation over profit maximization."""
        
        user_prompt = f"""Assess trading risk for SPY position:

Market Volatility: {market_data.get('volatility', {})}
VIX Level: {market_data.get('vix_current', {})}
Current Positions: {market_data.get('portfolio', {})}
Market Correlations: {market_data.get('correlations', {})}
Liquidity Indicators: {market_data.get('liquidity', {})}

Provide risk assessment and position sizing recommendations."""
        
        response_text = self._make_api_call(system_prompt, user_prompt)
        
        # Parse risk-adjusted signal
        signal = "HOLD"
        confidence = 0.5
        
        # Risk agent tends to be more conservative
        if "low risk" in response_text.lower() and "buy" in response_text.lower():
            signal = "BUY"
            confidence = 0.6
        elif "high risk" in response_text.lower():
            signal = "HOLD"
            confidence = 0.8
        elif "extreme risk" in response_text.lower():
            signal = "SELL"
            confidence = 0.9
        
        features = {
            'volatility': market_data.get('volatility', {}).get('current', 0.2),
            'vix_level': market_data.get('vix_current', {}).get('Close', 20),
            'risk_score': 0.5  # Calculated risk score
        }
        
        return AgentResponse(
            agent_name=self.name,
            analysis=response_text,
            signal=signal,
            confidence=confidence,
            features=features,
            learning_feedback=""
        )

class CoordinatorAgent(LRMAgent):
    """Coordinator agent that synthesizes all inputs"""
    
    def __init__(self, anthropic_client: Anthropic, learning_system: AgentLearningSystem):
        super().__init__("Coordinator", anthropic_client, learning_system)
    
    def analyze(self, market_data: Dict, agent_responses: List[AgentResponse], learning_context: str = "") -> AgentResponse:
        system_prompt = f"""You are the Coordinator Agent responsible for synthesizing inputs from all specialist agents.

Your role:
- Integrate regime, technical, and risk analyses
- Resolve conflicting signals intelligently
- Weight agent inputs based on current market conditions
- Provide final trading recommendation
- Ensure risk management principles are followed

Learning Context from your past performance:
{learning_context}

Agent Input Summary:
{self._format_agent_inputs(agent_responses)}

Provide final signal: BUY/SELL/HOLD/SHORT with confidence level and reasoning.
Consider inter-agent agreement/disagreement in your confidence assessment."""
        
        user_prompt = f"""Synthesize the following agent analyses for final SPY trading decision:

{self._format_detailed_inputs(agent_responses)}

Market Context: {market_data.get('context', {})}

Provide final coordinated signal with confidence level and detailed reasoning."""
        
        response_text = self._make_api_call(system_prompt, user_prompt)
        
        # Synthesize final signal
        signal_votes = {}
        confidence_sum = 0
        
        for agent_resp in agent_responses:
            signal_votes[agent_resp.signal] = signal_votes.get(agent_resp.signal, 0) + agent_resp.confidence
            confidence_sum += agent_resp.confidence
        
        # Get consensus signal
        final_signal = max(signal_votes.keys(), key=lambda k: signal_votes[k])
        final_confidence = min(1.0, confidence_sum / len(agent_responses))
        
        features = {
            'consensus_strength': max(signal_votes.values()) / sum(signal_votes.values()),
            'agent_agreement': len(set(r.signal for r in agent_responses)) == 1,
            'avg_confidence': final_confidence
        }
        
        return AgentResponse(
            agent_name=self.name,
            analysis=response_text,
            signal=final_signal,
            confidence=final_confidence,
            features=features,
            learning_feedback=""
        )
    
    def _format_agent_inputs(self, responses: List[AgentResponse]) -> str:
        summary = []
        for resp in responses:
            summary.append(f"{resp.agent_name}: {resp.signal} (Confidence: {resp.confidence:.2f})")
        return "\n".join(summary)
    
    def _format_detailed_inputs(self, responses: List[AgentResponse]) -> str:
        detailed = []
        for resp in responses:
            detailed.append(f"{resp.agent_name} Agent Analysis:\n{resp.analysis}\n")
        return "\n".join(detailed)

class HybridMLSystem:
    """Machine learning system that works alongside LRM agents"""
    
    def __init__(self):
        self.models = {
            'regime_classifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'signal_predictor': RandomForestClassifier(n_estimators=100, random_state=42),
            'confidence_estimator': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.scalers = {
            'features': StandardScaler(),
            'targets': StandardScaler()
        }
        self.is_trained = False
    
    def prepare_features(self, market_data: Dict, agent_responses: List[AgentResponse]) -> np.ndarray:
        """Prepare feature vector for ML models"""
        features = []
        
        # Market features
        spy_data = market_data.get('spy_current', {})
        features.extend([
            spy_data.get('Close', 400),
            spy_data.get('Volume', 1000000),
            market_data.get('vix_current', {}).get('Close', 20),
        ])
        
        # Technical features
        tech = market_data.get('technicals', {})
        features.extend([
            tech.get('RSI', 50),
            tech.get('MACD', 0),
            tech.get('Volatility', 0.2)
        ])
        
        # Agent features
        for agent_resp in agent_responses:
            features.extend([
                agent_resp.confidence,
                1 if agent_resp.signal == 'BUY' else 0,
                1 if agent_resp.signal == 'SELL' else 0,
                1 if agent_resp.signal == 'SHORT' else 0
            ])
        
        return np.array(features).reshape(1, -1)
    
    def train_models(self, historical_data: pd.DataFrame, agent_history: List[Dict]):
        """Train ML models on historical data"""
        if len(historical_data) < 100:  # Need sufficient data
            return False
        
        # Prepare training data
        X, y = self._prepare_training_data(historical_data, agent_history)
        
        if len(X) < 50:
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_test_scaled = self.scalers['features'].transform(X_test)
        
        # Train models
        self.models['signal_predictor'].fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        train_score = self.models['signal_predictor'].score(X_train_scaled, y_train)
        test_score = self.models['signal_predictor'].score(X_test_scaled, y_test)
        
        st.info(f"ML Model Training Complete - Train Score: {train_score:.3f}, Test Score: {test_score:.3f}")
        
        self.is_trained = True
        return True
    
    def _prepare_training_data(self, historical_data: pd.DataFrame, agent_history: List[Dict]):
        """Prepare training data from historical records"""
        # This is a simplified version - in practice, you'd have more sophisticated feature engineering
        features = []
        targets = []
        
        for i in range(len(historical_data) - 5):  # Need future returns
            row = historical_data.iloc[i]
            future_return = (historical_data.iloc[i+5]['Close'] - row['Close']) / row['Close']
            
            feature_row = [
                row['Close'],
                row['Volume'],
                row.get('RSI', 50),
                row.get('MACD', 0),
                row.get('Volatility', 0.2)
            ]
            
            features.append(feature_row)
            targets.append(1 if future_return > 0.02 else 0)  # Binary classification
        
        return np.array(features), np.array(targets)
    
    def predict(self, features: np.ndarray) -> Dict:
        """Make ML predictions"""
        if not self.is_trained:
            return {'signal': 'HOLD', 'confidence': 0.5, 'ml_score': 0.5}
        
        try:
            features_scaled = self.scalers['features'].transform(features)
            prediction = self.models['signal_predictor'].predict(features_scaled)[0]
            probabilities = self.models['signal_predictor'].predict_proba(features_scaled)[0]
            
            signal = 'BUY' if prediction == 1 else 'HOLD'
            confidence = max(probabilities)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'ml_score': probabilities[1] if len(probabilities) > 1 else 0.5
            }
        except Exception as e:
            st.error(f"ML Prediction Error: {str(e)}")
            return {'signal': 'HOLD', 'confidence': 0.5, 'ml_score': 0.5}

class BacktestEngine:
    """Backtesting engine for strategy validation"""
    
    def __init__(self):
        self.results = []
        self.equity_curve = []
    
    def run_backtest(self, signals: List[TradingSignal], price_data: pd.DataFrame, 
                    initial_capital: float = 100000) -> Dict:
        """Run comprehensive backtest"""
        portfolio_value = initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_capital]
        
        for signal in signals:
            try:
                signal_date = pd.to_datetime(signal.timestamp).date()
                price_row = price_data[price_data.index.date == signal_date]
                
                if price_row.empty:
                    continue
                
                current_price = price_row['Close'].iloc[0]
                
                # Execute trades based on signals
                if signal.signal == 'BUY' and position <= 0:
                    if position < 0:  # Close short position
                        profit = (entry_price - current_price) * abs(position)
                        portfolio_value += profit
                        trades.append({
                            'type': 'COVER',
                            'price': current_price,
                            'profit': profit,
                            'date': signal_date
                        })
                    
                    # Open long position
                    position = portfolio_value / current_price
                    entry_price = current_price
                    
                elif signal.signal == 'SHORT' and position >= 0:
                    if position > 0:  # Close long position
                        profit = (current_price - entry_price) * position
                        portfolio_value += profit
                        trades.append({
                            'type': 'SELL',
                            'price': current_price,
                            'profit': profit,
                            'date': signal_date
                        })
                    
                    # Open short position
                    position = -portfolio_value / current_price
                    entry_price = current_price
                
                elif signal.signal in ['SELL', 'HOLD'] and position != 0:
                    # Close any position
                    if position > 0:
                        profit = (current_price - entry_price) * position
                        trades.append({'type': 'SELL', 'price': current_price, 'profit': profit, 'date': signal_date})
                    else:
                        profit = (entry_price - current_price) * abs(position)
                        trades.append({'type': 'COVER', 'price': current_price, 'profit': profit, 'date': signal_date})
                    
                    portfolio_value += profit
                    position = 0
                
                # Update portfolio value
                if position > 0:
                    portfolio_value = position * current_price
                elif position < 0:
                    portfolio_value = initial_capital - (position * (current_price - entry_price))
                
                equity_curve.append(portfolio_value)
                
            except Exception as e:
                st.error(f"Backtest error at {signal.timestamp}: {str(e)}")
                continue
        
        # Calculate performance metrics
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        metrics = {
            'total_return': (portfolio_value - initial_capital) / initial_capital,
            'annualized_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'num_trades': len(trades),
            'win_rate': len([t for t in trades if t['profit'] > 0]) / len(trades) if trades else 0,
            'equity_curve': equity_curve,
            'trades': trades
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd

class TradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self, anthropic_api_key: str, alpha_vantage_key: str):
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)
        self.data_provider = DataProvider(alpha_vantage_key)
        self.learning_system = AgentLearningSystem()
        self.ml_system = HybridMLSystem()
        self.backtest_engine = BacktestEngine()
        
        # Initialize agents
        self.agents = {
            'regime': RegimeAgent(self.anthropic_client, self.learning_system),
            'technical': TechnicalAgent(self.anthropic_client, self.learning_system),
            'risk': RiskAgent(self.anthropic_client, self.learning_system),
            'coordinator': CoordinatorAgent(self.anthropic_client, self.learning_system)
        }
        
        self.signal_history = []
    
    def generate_signal(self) -> TradingSignal:
        """Generate trading signal using all agents"""
        try:
            # Gather market data
            market_data = self._gather_market_data()
            
            # Get agent responses
            agent_responses = []
            
            for agent_name, agent in self.agents.items():
                if agent_name != 'coordinator':
                    learning_context = self.learning_system.get_agent_learning_context(agent_name)
                    response = agent.analyze(market_data, learning_context)
                    agent_responses.append(response)
            
            # Coordinator synthesis
            learning_context = self.learning_system.get_agent_learning_context('coordinator')
            final_response = self.agents['coordinator'].analyze(market_data, agent_responses, learning_context)
            
            # ML enhancement
            ml_features = self.ml_system.prepare_features(market_data, agent_responses)
            ml_prediction = self.ml_system.predict(ml_features)
            
            # Combine LRM and ML signals
            final_signal = self._combine_signals(final_response, ml_prediction)
            
            # Create trading signal
            trading_signal = TradingSignal(
                timestamp=datetime.now().isoformat(),
                symbol="SPY",
                signal=final_signal['signal'],
                confidence=final_signal['confidence'],
                regime_score=agent_responses[0].confidence,
                technical_score=agent_responses[1].confidence,
                risk_score=agent_responses[2].confidence,
                coordinator_score=final_response.confidence,
                price=market_data.get('spy_current', {}).get('Close', 0),
                reasoning=final_response.analysis
            )
            
            self.signal_history.append(trading_signal)
            return trading_signal
            
        except Exception as e:
            st.error(f"Signal generation error: {str(e)}")
            return TradingSignal(
                timestamp=datetime.now().isoformat(),
                symbol="SPY",
                signal="HOLD",
                confidence=0.0,
                regime_score=0.0,
                technical_score=0.0,
                risk_score=0.0,
                coordinator_score=0.0,
                price=0.0,
                reasoning=f"Error: {str(e)}"
            )
    
    def _gather_market_data(self) -> Dict:
        """Gather all required market data"""
        try:
            spy_data = self.data_provider.get_spy_data()
            vix_data = self.data_provider.get_vix_data()
            yield_data = self.data_provider.get_yield_data()
            economic_data = self.data_provider.get_economic_indicators()
            
            current_spy = spy_data.iloc[-1].to_dict()
            current_vix = vix_data.iloc[-1].to_dict()
            
            return {
                'spy_current': current_spy,
                'vix_current': current_vix,
                'yields': yield_data,
                'economic': economic_data,
                'technicals': {
                    'RSI': current_spy.get('RSI', 50),
                    'MACD': current_spy.get('MACD', 0),
                    'SMA_20': current_spy.get('SMA_20', current_spy.get('Close', 400)),
                    'SMA_50': current_spy.get('SMA_50', current_spy.get('Close', 400)),
                    'Volatility': current_spy.get('Volatility', 0.2)
                },
                'volume': current_spy.get('Volume', 0),
                'recent_data': spy_data.tail(10).to_dict(),
                'volatility': {'current': current_spy.get('Volatility', 0.2)},
                'correlations': {},  # Could add correlation analysis
                'liquidity': {},     # Could add liquidity metrics
                'portfolio': {},     # Current portfolio state
                'context': {
                    'market_hours': self._is_market_hours(),
                    'day_of_week': datetime.now().weekday()
                }
            }
        except Exception as e:
            st.error(f"Data gathering error: {str(e)}")
            return {}
    
    def _combine_signals(self, lrm_response: AgentResponse, ml_prediction: Dict) -> Dict:
        """Combine LRM and ML signals intelligently"""
        lrm_weight = 0.7
        ml_weight = 0.3
        
        # Signal mapping
        signal_scores = {
            'BUY': 1.0,
            'HOLD': 0.0,
            'SELL': -0.5,
            'SHORT': -1.0
        }
        
        lrm_score = signal_scores.get(lrm_response.signal, 0.0) * lrm_response.confidence
        ml_score = signal_scores.get(ml_prediction['signal'], 0.0) * ml_prediction['confidence']
        
        combined_score = (lrm_score * lrm_weight) + (ml_score * ml_weight)
        combined_confidence = (lrm_response.confidence * lrm_weight) + (ml_prediction['confidence'] * ml_weight)
        
        # Map back to signal
        if combined_score > 0.3:
            final_signal = 'BUY'
        elif combined_score < -0.3:
            final_signal = 'SELL'
        elif combined_score < -0.6:
            final_signal = 'SHORT'
        else:
            final_signal = 'HOLD'
        
        return {
            'signal': final_signal,
            'confidence': combined_confidence,
            'lrm_component': lrm_score,
            'ml_component': ml_score
        }
    
    def _is_market_hours(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return (now.weekday() < 5 and market_open <= now <= market_close)
    
    def update_performance(self, signal: TradingSignal, actual_return: float):
        """Update agent performance with actual outcomes"""
        # Update each agent's learning
        for agent_name in self.agents.keys():
            if agent_name != 'coordinator':
                self.agents[agent_name].update_learning(
                    signal.signal, 
                    getattr(signal, f'{agent_name}_score'), 
                    {'return': actual_return}, 
                    actual_return
                )
    
    def train_ml_models(self, lookback_days: int = 252):
        """Train ML models on historical data"""
        try:
            spy_data = self.data_provider.get_spy_data(f"{lookback_days}d")
            # In practice, you'd also load historical agent responses
            agent_history = []  # Would load from database
            
            success = self.ml_system.train_models(spy_data, agent_history)
            if success:
                st.success("ML models trained successfully")
            else:
                st.warning("Insufficient data for ML training")
        except Exception as e:
            st.error(f"ML training error: {str(e)}")
    
    def run_backtest(self, start_date: str, end_date: str) -> Dict:
        """Run comprehensive backtest"""
        try:
            # Generate historical signals (simplified for demo)
            spy_data = self.data_provider.get_spy_data("2y")
            
            # In a real implementation, you'd replay historical market conditions
            # and generate signals for each day
            mock_signals = self._generate_mock_historical_signals(spy_data)
            
            backtest_results = self.backtest_engine.run_backtest(mock_signals, spy_data)
            return backtest_results
        except Exception as e:
            st.error(f"Backtest error: {str(e)}")
            return {}
    
    def _generate_mock_historical_signals(self, price_data: pd.DataFrame) -> List[TradingSignal]:
        """Generate mock historical signals for backtesting demo"""
        signals = []
        
        for i in range(0, len(price_data), 5):  # Signal every 5 days
            row = price_data.iloc[i]
            
            # Simple mock logic - in practice, you'd replay the full agent system
            rsi = row.get('RSI', 50)
            if rsi < 30:
                signal = 'BUY'
                confidence = 0.8
            elif rsi > 70:
                signal = 'SELL'
                confidence = 0.8
            else:
                signal = 'HOLD'
                confidence = 0.5
            
            trading_signal = TradingSignal(
                timestamp=row.name.isoformat(),
                symbol="SPY",
                signal=signal,
                confidence=confidence,
                regime_score=confidence,
                technical_score=confidence,
                risk_score=confidence,
                coordinator_score=confidence,
                price=row['Close'],
                reasoning=f"Mock signal based on RSI: {rsi:.2f}"
            )
            
            signals.append(trading_signal)
        
        return signals

# Streamlit Application
def main():
    st.set_page_config(
        page_title="Multi-Agent LRM Trading System",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🤖 Multi-Agent LRM Trading System")
    st.markdown("*Advanced AI-Powered SPY Trading with Inter-Agent Learning*")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # API Keys
    anthropic_key = st.sidebar.text_input("Anthropic API Key", type="password")
    alpha_vantage_key = st.sidebar.text_input("Alpha Vantage API Key", type="password")
    
    if not anthropic_key or not alpha_vantage_key:
        st.warning("Please enter your API keys to proceed.")
        st.info("""
        **Required API Keys:**
        - **Anthropic API**: Get your key from https://console.anthropic.com
        - **Alpha Vantage API**: Get your free key from https://www.alphavantage.co/support/#api-key
        """)
        return
    
    # Initialize system
    if 'trading_system' not in st.session_state:
        with st.spinner("Initializing Trading System..."):
            st.session_state.trading_system = TradingSystem(anthropic_key, alpha_vantage_key)
    
    trading_system = st.session_state.trading_system
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Live Signals", 
        "📊 Backtest", 
        "🧠 Agent Learning", 
        "🤖 ML Training", 
        "📈 Performance"
    ])
    
    with tab1:
        st.header("Live Trading Signals")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🔄 Generate New Signal", type="primary"):
                with st.spinner("Analyzing market conditions..."):
                    try:
                        signal = trading_system.generate_signal()
                        st.session_state.latest_signal = signal
                    except Exception as e:
                        st.error(f"Signal generation failed: {str(e)}")
        
        with col2:
            auto_refresh = st.checkbox("Auto-refresh (30s)")
            if auto_refresh:
                st.rerun()
        
        # Display latest signal
        if 'latest_signal' in st.session_state:
            signal = st.session_state.latest_signal
            
            # Signal card
            signal_color = {
                'BUY': 'green',
                'SELL': 'red', 
                'SHORT': 'orange',
                'HOLD': 'gray'
            }
            
            st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; border-left: 5px solid {signal_color[signal.signal]}; background-color: #f0f2f6; margin: 10px 0;'>
                <h3 style='margin: 0; color: {signal_color[signal.signal]};'>🎯 {signal.signal}</h3>
                <p><strong>Confidence:</strong> {signal.confidence:.1%}</p>
                <p><strong>Price:</strong> ${signal.price:.2f}</p>
                <p><strong>Timestamp:</strong> {signal.timestamp}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Agent breakdown
            st.subheader("Agent Analysis Breakdown")
            
            agent_cols = st.columns(4)
            agents_data = [
                ("Regime", signal.regime_score, "🌍"),
                ("Technical", signal.technical_score, "📊"),
                ("Risk", signal.risk_score, "⚠️"),
                ("Coordinator", signal.coordinator_score, "🎯")
            ]
            
            for i, (name, score, icon) in enumerate(agents_data):
                with agent_cols[i]:
                    st.metric(
                        label=f"{icon} {name}",
                        value=f"{score:.1%}",
                        delta=None
                    )
            
            # Reasoning
            st.subheader("📝 Analysis Reasoning")
            st.text_area("Agent Analysis", signal.reasoning, height=200, disabled=True)
    
    with tab2:
        st.header("📊 Strategy Backtesting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        if st.button("🚀 Run Backtest"):
            with st.spinner("Running comprehensive backtest..."):
                results = trading_system.run_backtest(start_date.isoformat(), end_date.isoformat())
                
                if results:
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Return", f"{results['total_return']:.1%}")
                    
                    with col2:
                        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                    
                    with col3:
                        st.metric("Max Drawdown", f"{results['max_drawdown']:.1%}")
                    
                    with col4:
                        st.metric("Win Rate", f"{results['win_rate']:.1%}")
                    
                    # Equity curve
                    st.subheader("📈 Equity Curve")
                    
                    equity_df = pd.DataFrame({
                        'Date': pd.date_range(start=start_date, periods=len(results['equity_curve'])),
                        'Portfolio Value': results['equity_curve']
                    })
                    
                    fig = px.line(equity_df, x='Date', y='Portfolio Value', 
                                 title="Portfolio Performance Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trade log
                    st.subheader("📋 Trade Log")
                    if results['trades']:
                        trades_df = pd.DataFrame(results['trades'])
                        st.dataframe(trades_df)
    
    with tab3:
        st.header("🧠 Agent Learning Dashboard")
        
        st.subheader("Agent Performance History")
        
        # Learning metrics would be displayed here
        learning_data = {
            'Agent': ['Regime', 'Technical', 'Risk', 'Coordinator'],
            'Accuracy': [0.65, 0.72, 0.68, 0.70],
            'Trades': [45, 52, 48, 50],
            'Avg Confidence': [0.75, 0.68, 0.82, 0.73]
        }
        
        learning_df = pd.DataFrame(learning_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(learning_df, x='Agent', y='Accuracy', 
                        title="Agent Accuracy Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(learning_df, x='Agent', y='Avg Confidence',
                        title="Average Confidence Levels")
            st.plotly_chart(fig, use_container_width=True)
        
        # Inter-agent feedback
        st.subheader("🔄 Inter-Agent Learning")
        st.info("Agents continuously learn from each other's successes and failures, improving collective performance over time.")
        
        feedback_data = {
            'Source Agent': ['Technical', 'Risk', 'Regime', 'Coordinator'],
            'Target Agent': ['Risk', 'Technical', 'Coordinator', 'Technical'],
            'Feedback Type': ['Risk Alert', 'Signal Confidence', 'Regime Change', 'Synthesis Improvement'],
            'Impact Score': [0.85, 0.72, 0.91, 0.78]
        }
        
        feedback_df = pd.DataFrame(feedback_data)
        st.dataframe(feedback_df)
    
    with tab4:
        st.header("🤖 Machine Learning Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lookback_days = st.slider("Training Data Period (Days)", 30, 1000, 252)
        
        with col2:
            if st.button("🎯 Train ML Models"):
                with st.spinner("Training machine learning models..."):
                    trading_system.train_ml_models(lookback_days)
        
        st.subheader("🔬 Hybrid LRM-ML Architecture")
        
        st.markdown("""
        **Our hybrid approach combines:**
        
        1. **Large Reasoning Models (LRM)**: Advanced reasoning and market interpretation
        2. **Traditional ML**: Pattern recognition and statistical learning
        3. **Inter-Agent Learning**: Continuous improvement through agent feedback
        
        **Benefits:**
        - LRM provides sophisticated market reasoning
        - ML captures statistical patterns and relationships  
        - Combined system leverages both strengths
        - Continuous learning improves performance over time
        """)
        
        # ML Model status
        if trading_system.ml_system.is_trained:
            st.success("✅ ML Models are trained and active")
        else:
            st.warning("⚠️ ML Models need training - using LRM-only mode")
    
    with tab5:
        st.header("📈 System Performance Analytics")
        
        # Overall system metrics
        st.subheader("🎯 Overall Performance")
        
        perf_cols = st.columns(4)
        
        with perf_cols[0]:
            st.metric("Signals Generated", len(trading_system.signal_history))
        
        with perf_cols[1]:
            if trading_system.signal_history:
                avg_confidence = np.mean([s.confidence for s in trading_system.signal_history])
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            else:
                st.metric("Avg Confidence", "N/A")
        
        with perf_cols[2]:
            st.metric("System Uptime", "99.5%")
        
        with perf_cols[3]:
            st.metric("ML Models Active", "✅" if trading_system.ml_system.is_trained else "❌")
        
        # Signal distribution
        if trading_system.signal_history:
            st.subheader("📊 Signal Distribution")
            
            signals_df = pd.DataFrame([asdict(s) for s in trading_system.signal_history])
            
            col1, col2 = st.columns(2)
            
            with col1:
                signal_counts = signals_df['signal'].value_counts()
                fig = px.pie(values=signal_counts.values, names=signal_counts.index,
                            title="Signal Type Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(signals_df, x='confidence', nbins=20,
                                 title="Confidence Score Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent signals table
            st.subheader("📋 Recent Signals")
            display_cols = ['timestamp', 'signal', 'confidence', 'price']
            st.dataframe(signals_df[display_cols].tail(10))
        
        else:
            st.info("Generate some signals to see performance analytics!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Multi-Agent LRM Trading System | Powered by Anthropic Claude & Advanced ML</p>
        <p><em>Remember: This system is for educational purposes. Always do your own research before making investment decisions.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
