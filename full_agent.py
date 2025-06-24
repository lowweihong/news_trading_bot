import os
import requests
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template
import threading
import queue
import random
import numpy as np
from transformers import pipeline

# Load environment variables
load_dotenv()
API_KEY = os.getenv("SUPRA_API_KEY")
REST_BASE_URL = "https://prod-kline-rest.supra.com"
HISTORY_ENDPOINT = "/history"
LATEST_ENDPOINT = "/latest"

class FullAITradingAgent:
    def __init__(self):
        # Only process ETH for speed
        self.assets = {
            'eth_usd': 'Ethereum'
        }
        self.portfolio = {k: 0 for k in self.assets}
        self.portfolio_value = 10000
        self.trade_log = []
        self.signals = []
        self.sentiment_scores = []
        self.price_history = {}
        self.news_history = []
        
        # Enhanced trading parameters
        self.sentiment_threshold = 0.15
        self.price_change_threshold = 1.0  # 1% price change
        self.confidence_threshold = 0.3
        # Initialize Hugging Face financial sentiment model
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        )

    def fetch_historical_prices(self, trading_pair, days=30):
        """Fetch 30 days of historical price data"""
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        url = f"{REST_BASE_URL}{HISTORY_ENDPOINT}"
        headers = {"x-api-key": API_KEY}
        params = {
            "trading_pair": trading_pair,
            "startDate": start_time,
            "endDate": end_time,
            "interval": 60  # 1-hour intervals
        }
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                raw_data = response.json()
                # Supra may return a list directly or a dict with 'data' key
                if isinstance(raw_data, list):
                    data = [
                        {
                            **item,
                            'open': float(item['open']),
                            'high': float(item['high']),
                            'low': float(item['low']),
                            'close': float(item['close'])
                        }
                        for item in raw_data
                        if all(k in item for k in ['open', 'high', 'low', 'close'])
                    ]
                elif isinstance(raw_data, dict) and 'data' in raw_data:
                    data = [
                        {
                            **item,
                            'open': float(item['open']),
                            'high': float(item['high']),
                            'low': float(item['low']),
                            'close': float(item['close'])
                        }
                        for item in raw_data['data']
                        if all(k in item for k in ['open', 'high', 'low', 'close'])
                    ]
                else:
                    print(f"⚠️ Unexpected Supra history API response: {raw_data}")
                    data = []
                self.price_history[trading_pair] = data
                print(f"✅ Fetched {len(data)} historical data points for {trading_pair}")
                return data
            else:
                print(f"❌ API Error for {trading_pair}: {response.status_code}")
                return self._generate_historical_data(trading_pair, days)
        except Exception as e:
            print(f"Error fetching historical prices for {trading_pair}: {e}")
            return self._generate_historical_data(trading_pair, days)

    def _generate_historical_data(self, trading_pair, days):
        """Generate realistic historical data for demo"""
        base_prices = {
            'btc_usd': 48500,
            'eth_usd': 3200,
            'supra_usd': 0.85,
            'sol_usd': 95,
            'ada_usd': 0.45
        }
        
        base_price = base_prices.get(trading_pair, 100)
        data = []
        
        for i in range(days * 24):  # Hourly data
            timestamp = int(time.time() * 1000) - (i * 60 * 60 * 1000)
            
            # Simulate realistic price movements
            if i == 0:
                price = base_price
            else:
                # Add some trend and volatility
                trend = np.random.normal(0, 0.001)  # Small trend
                volatility = np.random.normal(0, 0.005)  # Price volatility
                price = data[-1]['close'] * (1 + trend + volatility)
            
            high = price * (1 + abs(np.random.normal(0, 0.003)))
            low = price * (1 - abs(np.random.normal(0, 0.003)))
            volume = random.randint(1000000, 10000000)
            
            data.append({
                'timestamp': timestamp,
                'open': price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        return data[::-1]  # Reverse to chronological order

    def fetch_news(self):
        """Fetch only 10 news articles (1 page) for speed"""
        all_news = []
        url = "https://api.theblockbeats.news/v1/open-api/open-information"
        params = {'size': 100, 'page': 1, 'lang': 'en'}
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                news_data = response.json()
                news_items = news_data.get('data', {}).get('data', [])
                all_news.extend(news_items)
                print(f"✅ Fetched {len(news_items)} news articles from page 1")
            else:
                print(f"❌ News API error on page 1: {response.status_code}")
        except Exception as e:
            print(f"Error fetching news page 1: {e}")
        self.news_history = all_news
        print(f"📰 Total news articles fetched: {len(all_news)}")
        return all_news

    def analyze_sentiment(self, text):
        """Analyze sentiment using DistilRoberta financial model and assess market impact"""
        try:
            result = self.sentiment_model(text[:512])[0]  # Truncate to 512 tokens
            label = result['label'].lower()
            score = result['score']
            # Map to a polarity score for compatibility
            if label == 'positive':
                sentiment_score = score
            elif label == 'negative':
                sentiment_score = -score
            else:
                sentiment_score = 0
            market_impact = self._assess_market_impact(text)
            return {
                'sentiment': sentiment_score,
                'confidence': score,
                'label': label,
                'market_impact': market_impact
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {'sentiment': 0, 'confidence': 0, 'label': 'neutral', 'market_impact': 'low'}

    def _assess_market_impact(self, text):
        """Assess potential market impact of news"""
        impact_keywords = {
            'high': ['regulation', 'sec', 'ban', 'adoption', 'institutional', 'etf', 'partnership', 'launch'],
            'medium': ['upgrade', 'hack', 'outage', 'announcement', 'update'],
            'low': ['community', 'development', 'update', 'news']
        }
        
        text_lower = text.lower()
        for impact_level, keywords in impact_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return impact_level
        
        return 'low'

    def calculate_technical_indicators(self, prices):
        """Calculate simple technical indicators"""
        if len(prices) < 20:
            return {}
        
        closes = [p['close'] for p in prices]
        
        # Simple moving averages
        sma_7 = np.mean(closes[-7:])
        sma_20 = np.mean(closes[-20:])
        
        # Price momentum
        momentum = (closes[-1] - closes[-5]) / closes[-5] * 100
        
        # Volatility
        volatility = np.std(closes[-20:]) / np.mean(closes[-20:]) * 100
        
        return {
            'sma_7': sma_7,
            'sma_20': sma_20,
            'momentum': momentum,
            'volatility': volatility,
            'trend': 'bullish' if sma_7 > sma_20 else 'bearish'
        }

    def generate_signal(self, asset, sentiment, price_data, technical_data):
        """Enhanced signal generation with multiple factors"""
        sentiment_score = sentiment['sentiment']
        confidence = sentiment['confidence']
        price_change = price_data.get('price_change_24h', 0)
        market_impact = sentiment['market_impact']
        
        # Technical analysis
        trend = technical_data.get('trend', 'neutral')
        momentum = technical_data.get('momentum', 0)
        volatility = technical_data.get('volatility', 0)
        
        # Signal scoring
        buy_score = 0
        sell_score = 0
        
        # Sentiment factors
        if sentiment_score > self.sentiment_threshold:
            buy_score += sentiment_score * 0.4
        elif sentiment_score < -self.sentiment_threshold:
            sell_score += abs(sentiment_score) * 0.4
        
        # Price momentum factors
        if price_change < -self.price_change_threshold and sentiment_score > 0:
            buy_score += 0.3  # Buy the dip with positive sentiment
        elif price_change > self.price_change_threshold and sentiment_score < 0:
            sell_score += 0.3  # Sell the rally with negative sentiment
        
        # Technical factors
        if trend == 'bullish':
            buy_score += 0.2
        elif trend == 'bearish':
            sell_score += 0.2
        
        # Market impact factors
        impact_multiplier = {'high': 1.5, 'medium': 1.0, 'low': 0.5}
        buy_score *= impact_multiplier.get(market_impact, 1.0)
        sell_score *= impact_multiplier.get(market_impact, 1.0)
        
        # Generate final signal
        if buy_score > sell_score and buy_score > 0.3:
            reason = f"Strong buy signal: Positive sentiment ({sentiment_score:.2f}), {trend} trend, {market_impact} impact"
            return 'BUY', reason, buy_score
        elif sell_score > buy_score and sell_score > 0.3:
            reason = f"Strong sell signal: Negative sentiment ({sentiment_score:.2f}), {trend} trend, {market_impact} impact"
            return 'SELL', reason, sell_score
        else:
            return 'HOLD', 'Insufficient signal strength', 0

    def simulate_trading(self):
        """Enhanced trading simulation with more comprehensive analysis"""
        print("🔄 Starting enhanced trading simulation...")
        
        self.signals.clear()
        self.trade_log.clear()
        self.sentiment_scores.clear()
        
        # Fetch news first
        news_items = self.fetch_news()
        if not news_items:
            print("❌ No news available for analysis")
            return
        
        # Process each asset
        for asset in self.assets:
            print(f"📊 Analyzing {asset}...")
            
            # Get historical data
            prices = self.fetch_historical_prices(asset, days=30)
            if not prices:
                continue
            
            # Calculate technical indicators
            technical_data = self.calculate_technical_indicators(prices)
            
            # Get recent price change
            if len(prices) >= 2:
                last_price = prices[-1]['close']
                prev_price = prices[-2]['close']
                price_change = ((last_price - prev_price) / prev_price) * 100
            else:
                price_change = 0
                last_price = prices[0]['close']
            
            # Find relevant news for this asset
            asset_news = []
            for news in news_items:
                text = f"{news.get('title', '')}. {news.get('description', '')}. {news.get('content', '')[:300]}"
                if (self.assets[asset].lower() in text.lower() or 
                    asset.replace('_', '').lower() in text.lower() or
                    asset.split('_')[0].upper() in text.upper()):
                    asset_news.append(news)
            
            print(f"   Found {len(asset_news)} relevant news articles")
            
            # Analyze each relevant news article
            for news in asset_news:
                text = f"{news.get('title', '')}. {news.get('description', '')}. {news.get('content', '')[:300]}"
                sentiment = self.analyze_sentiment(text)
                
                # Generate signal
                action, reason, signal_strength = self.generate_signal(
                    asset, sentiment, 
                    {'price_change_24h': price_change}, 
                    technical_data
                )
                
                # Create signal record
                signal = {
                    'asset': asset,
                    'action': action,
                    'reason': reason,
                    'signal_strength': signal_strength,
                    'confidence': sentiment['confidence'],
                    'price_change': price_change,
                    'current_price': last_price,
                    'technical_trend': technical_data.get('trend', 'neutral'),
                    'market_impact': sentiment['market_impact'],
                    'timestamp': datetime.now().isoformat(),
                    'news_title': news.get('title', '')[:100] + "..." if len(news.get('title', '')) > 100 else news.get('title', '')
                }
                
                self.signals.append(signal)
                
                # Store sentiment data
                self.sentiment_scores.append({
                    'asset': asset,
                    'sentiment': sentiment['sentiment'],
                    'confidence': sentiment['confidence'],
                    'market_impact': sentiment['market_impact'],
                    'timestamp': datetime.now().isoformat(),
                    'news_title': news.get('title', '')[:100] + "..." if len(news.get('title', '')) > 100 else news.get('title', '')
                })
                
                # Execute trade if signal is strong enough
                if action in ['BUY', 'SELL'] and signal_strength > 0.4:
                    trade_amount = 1000  # Fixed trade size for demo
                    
                    trade = {
                        'asset': asset,
                        'action': action,
                        'price': last_price,
                        'amount': trade_amount,
                        'signal_strength': signal_strength,
                        'timestamp': datetime.now().isoformat(),
                        'reason': reason,
                        'news_title': news.get('title', '')[:100] + "..." if len(news.get('title', '')) > 100 else news.get('title', '')
                    }
                    
                    self.trade_log.append(trade)
                    
                    # Update portfolio (simplified)
                    if action == 'BUY':
                        self.portfolio[asset] += trade_amount / last_price
                        self.portfolio_value -= trade_amount
                    elif action == 'SELL' and self.portfolio[asset] > 0:
                        sell_value = self.portfolio[asset] * last_price
                        self.portfolio_value += sell_value
                        self.portfolio[asset] = 0
                    
                    print(f"   ✅ {action} signal: {reason}")
        
        print(f"🎯 Generated {len(self.signals)} signals, executed {len(self.trade_log)} trades")

    def get_dashboard_data(self):
        """Enhanced dashboard data with more metrics"""
        total_trades = len(self.trade_log)
        buy_signals = len([s for s in self.signals if s['action'] == 'BUY'])
        sell_signals = len([s for s in self.signals if s['action'] == 'SELL'])
        
        # Calculate average sentiment
        if self.sentiment_scores:
            avg_sentiment = sum([s['sentiment'] for s in self.sentiment_scores]) / len(self.sentiment_scores)
        else:
            avg_sentiment = 0
        
        return {
            'signals': self.signals[-15:],  # Last 15 signals
            'sentiments': self.sentiment_scores[-25:],  # Last 25 sentiment scores
            'portfolio': self.portfolio,
            'portfolio_value': self.portfolio_value,
            'trade_log': self.trade_log[-15:],  # Last 15 trades
            'assets': self.assets,
            'last_update': datetime.now().isoformat(),
            'summary': {
                'total_signals': len(self.signals),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'total_trades': total_trades,
                'avg_sentiment': avg_sentiment,
                'news_articles_analyzed': len(self.news_history)
            }
        }

# Flask app
app = Flask(__name__)
agent = FullAITradingAgent()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    return jsonify(agent.get_dashboard_data())

@app.route('/api/process')
def process():
    print("🔄 Processing news and generating trading signals...")
    agent.simulate_trading()
    return jsonify({
        'status': 'success', 
        'message': f'Processed {len(agent.news_history)} news articles and generated {len(agent.signals)} signals.'
    })

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'running',
        'portfolio_value': agent.portfolio_value,
        'total_signals': len(agent.signals),
        'total_trades': len(agent.trade_log)
    })

if __name__ == '__main__':
    print("🚀 Starting Enhanced AI Trading Agent Backend...")
    print("📊 Features:")
    print("   • 30-day historical data analysis")
    print("   • 30+ news articles processing")
    print("   • Multi-factor signal generation")
    print("   • Technical indicator analysis")
    print("   • Portfolio simulation")
    print("🌐 Dashboard available at: http://localhost:5002")
    app.run(debug=True, host='0.0.0.0', port=5002) 