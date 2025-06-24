#!/usr/bin/env python3
"""
Demo AI News Trading Agent - Perfect for 5-minute video presentation
Generates guaranteed trading signals using historical data and enhanced logic
"""

import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
from textblob import TextBlob
import yfinance as yf
from flask import Flask, render_template, jsonify
import threading
import queue
import os
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()
API_KEY = os.getenv("SUPRA_API_KEY")

REST_BASE_URL = "https://prod-kline-rest.supra.com"
HISTORY_ENDPOINT = "/history"
LATEST_ENDPOINT = "/latest"

class DemoNewsTradingAgent:
    def __init__(self, demo_mode=True):
        self.news_queue = queue.Queue()
        self.price_data = {}
        self.trading_signals = []
        self.sentiment_scores = []
        self.demo_mode = demo_mode
        
        # Demo-friendly trading parameters (more sensitive for demo)
        self.sentiment_threshold = 0.1  # Lower threshold for more signals
        self.price_change_threshold = 0.01  # 1% price change threshold
        self.confidence_threshold = 0.3  # Lower confidence requirement
        
        # Extended assets for more signal opportunities
        self.assets = {
            'btc_usd': 'Bitcoin',
            'eth_usd': 'Ethereum', 
            'supra_usd': 'Supra',
            'sol_usd': 'Solana',
            'ada_usd': 'Cardano',
            'dot_usd': 'Polkadot',
            'link_usd': 'Chainlink',
            'uni_usd': 'Uniswap',
            'aave_usd': 'Aave',
            'comp_usd': 'Compound'
        }
        
        # Demo historical data
        self.demo_news = [
            {
                'title': 'Bitcoin Surges Past $50,000 as Institutional Adoption Accelerates',
                'description': 'Major financial institutions announce Bitcoin investment strategies, driving massive price gains.',
                'content': 'Bitcoin has reached new heights as major banks and investment firms announce significant Bitcoin allocations. This represents a major milestone in cryptocurrency adoption.',
                'sentiment': 0.8,
                'assets': ['btc_usd']
            },
            {
                'title': 'Ethereum 2.0 Upgrade Faces Technical Challenges',
                'description': 'Developers encounter issues with the highly anticipated Ethereum upgrade, causing market uncertainty.',
                'content': 'The Ethereum 2.0 upgrade has hit some technical roadblocks, raising concerns about the timeline and implementation.',
                'sentiment': -0.6,
                'assets': ['eth_usd']
            },
            {
                'title': 'Supra Protocol Launches Revolutionary Oracle Network',
                'description': 'Supra introduces breakthrough technology for decentralized price feeds and data oracles.',
                'content': 'Supra has successfully launched its innovative oracle network, providing faster and more accurate price data to DeFi protocols.',
                'sentiment': 0.9,
                'assets': ['supra_usd']
            },
            {
                'title': 'Solana Network Experiences Temporary Outage',
                'description': 'High transaction volume causes brief network congestion on Solana blockchain.',
                'content': 'Solana faced network issues due to unprecedented transaction volume, though services have been restored.',
                'sentiment': -0.4,
                'assets': ['sol_usd']
            },
            {
                'title': 'DeFi Protocols See Record TVL Growth',
                'description': 'Total Value Locked in DeFi reaches new all-time highs as yield farming gains popularity.',
                'content': 'DeFi protocols are experiencing unprecedented growth with innovative yield farming strategies attracting billions in capital.',
                'sentiment': 0.7,
                'assets': ['aave_usd', 'comp_usd', 'uni_usd']
            },
            {
                'title': 'Regulatory Concerns Mount Over Stablecoin Markets',
                'description': 'Government officials express concerns about stablecoin market stability and regulation.',
                'content': 'Regulators are increasing scrutiny of stablecoin markets, potentially impacting the broader crypto ecosystem.',
                'sentiment': -0.5,
                'assets': ['btc_usd', 'eth_usd']
            },
            {
                'title': 'Chainlink Oracle Network Expands to New Blockchains',
                'description': 'Chainlink announces integration with multiple new blockchain networks.',
                'content': 'Chainlink continues its expansion by adding support for emerging blockchain platforms.',
                'sentiment': 0.6,
                'assets': ['link_usd']
            },
            {
                'title': 'Polkadot Parachain Auctions Generate Massive Interest',
                'description': 'Polkadot ecosystem sees unprecedented participation in parachain slot auctions.',
                'content': 'The Polkadot network is experiencing massive growth as projects compete for parachain slots.',
                'sentiment': 0.8,
                'assets': ['dot_usd']
            }
        ]
        
        # Demo price data
        self.demo_prices = {
            'btc_usd': {'price': 48500, 'change': 2.5},
            'eth_usd': {'price': 3200, 'change': -1.8},
            'supra_usd': {'price': 0.85, 'change': 15.2},
            'sol_usd': {'price': 95, 'change': -3.2},
            'ada_usd': {'price': 0.45, 'change': 1.1},
            'dot_usd': {'price': 6.8, 'change': 8.5},
            'link_usd': {'price': 12.5, 'change': 4.2},
            'uni_usd': {'price': 8.2, 'change': 2.8},
            'aave_usd': {'price': 85, 'change': 6.1},
            'comp_usd': {'price': 45, 'change': 3.9}
        }
        
    def fetch_crypto_news(self):
        """Fetch crypto news - demo mode uses historical data"""
        if self.demo_mode:
            print("🎭 Demo mode: Using historical news data")
            return self.demo_news
        else:
            # Real API call
            try:
                url = "https://api.theblockbeats.news/v1/open-api/open-information"
                params = {'size': 10, 'page': 1, 'lang': 'en'}
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    news_data = response.json()
                    if news_data.get('status') == 0:
                        return news_data.get('data', {}).get('data', [])
                return []
            except Exception as e:
                print(f"Error fetching news: {e}")
                return []
    
    def analyze_sentiment(self, text):
        """Analyze sentiment - demo mode uses predefined scores"""
        if self.demo_mode:
            # Use predefined sentiment for demo
            for news in self.demo_news:
                if news['title'] in text or news['description'] in text:
                    return {
                        'sentiment': news['sentiment'],
                        'subjectivity': 0.3,
                        'confidence': abs(news['sentiment']) * 0.8
                    }
            # Fallback to TextBlob
            return self._analyze_sentiment_textblob(text)
        else:
            return self._analyze_sentiment_textblob(text)
    
    def _analyze_sentiment_textblob(self, text):
        """Real sentiment analysis using TextBlob"""
        try:
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            subjectivity_score = blob.sentiment.subjectivity
            
            return {
                'sentiment': sentiment_score,
                'subjectivity': subjectivity_score,
                'confidence': abs(sentiment_score) * (1 - subjectivity_score)
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {'sentiment': 0, 'subjectivity': 0.5, 'confidence': 0}
    
    def get_price_data(self, symbol):
        """Get price data - demo mode uses predefined prices"""
        if self.demo_mode:
            print(f"🎭 Demo mode: Using historical price data for {symbol}")
            if symbol in self.demo_prices:
                price_data = self.demo_prices[symbol]
                return {
                    'symbol': symbol,
                    'price': price_data['price'],
                    'price_change_24h': price_data['change'],
                    'volume': random.randint(1000000, 10000000),
                    'market_cap': price_data['price'] * random.randint(1000000, 10000000),
                    'timestamp': datetime.now().isoformat()
                }
            return None
        else:
            # Real API call
            return self._get_real_price_data(symbol)
    
    def _get_real_price_data(self, symbol):
        """Real price data fetching"""
        try:
            # Try Supra API first
            if API_KEY:
                url = f"{REST_BASE_URL}{LATEST_ENDPOINT}"
                headers = {"x-api-key": API_KEY}
                params = {"trading_pair": symbol}
                response = requests.get(url, headers=headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    current_price = data.get('price', 0)
                    price_change = data.get('price_change_24h', 0)
                    
                    return {
                        'symbol': symbol,
                        'price': float(current_price) if current_price else 0,
                        'price_change_24h': float(price_change) if price_change else 0,
                        'volume': data.get('volume', 0),
                        'market_cap': data.get('market_cap', 0),
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Fallback to yfinance
            return self._get_yfinance_fallback(symbol)
                
        except Exception as e:
            print(f"Error fetching price data for {symbol}: {e}")
            return self._get_yfinance_fallback(symbol)
    
    def _get_yfinance_fallback(self, symbol):
        """Fallback method using yfinance"""
        try:
            if '_' in symbol:
                base, quote = symbol.split('_')
                yf_symbol = f"{base.upper()}-{quote.upper()}"
            else:
                yf_symbol = f"{symbol.upper()}-USD"
            
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            
            current_price = info.get('regularMarketPrice', 0)
            previous_close = info.get('previousClose', current_price)
            price_change = ((current_price - previous_close) / previous_close) * 100
            
            return {
                'symbol': symbol,
                'price': current_price,
                'price_change_24h': price_change,
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"YFinance fallback error for {symbol}: {e}")
            return None
    
    def generate_trading_signal(self, asset, sentiment_data, price_data):
        """Generate trading signal with enhanced logic for demo"""
        if not price_data:
            return None
            
        sentiment_score = sentiment_data['sentiment']
        confidence = sentiment_data['confidence']
        price_change = price_data['price_change_24h']
        
        signal = {
            'asset': asset,
            'timestamp': datetime.now().isoformat(),
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'price_change_24h': price_change,
            'current_price': price_data['price'],
            'action': 'HOLD',
            'reason': 'Insufficient confidence or neutral sentiment'
        }
        
        # Enhanced signal logic for demo
        if (abs(sentiment_score) > self.sentiment_threshold and 
            confidence > self.confidence_threshold):
            
            if sentiment_score > 0:  # Positive sentiment
                if price_change < -self.price_change_threshold:
                    signal['action'] = 'BUY'
                    signal['reason'] = f'Strong positive sentiment ({sentiment_score:.2f}) with price dip ({price_change:.2f}%) - Buy the dip!'
                elif price_change > self.price_change_threshold:
                    signal['action'] = 'BUY'
                    signal['reason'] = f'Positive sentiment ({sentiment_score:.2f}) with momentum ({price_change:.2f}%) - Trend following!'
                else:
                    signal['action'] = 'BUY'
                    signal['reason'] = f'Positive sentiment ({sentiment_score:.2f}) with stable price - Accumulation opportunity!'
            
            else:  # Negative sentiment
                if price_change > self.price_change_threshold:
                    signal['action'] = 'SELL'
                    signal['reason'] = f'Negative sentiment ({sentiment_score:.2f}) despite price rise ({price_change:.2f}%) - Potential reversal!'
                elif price_change < -self.price_change_threshold:
                    signal['action'] = 'SELL'
                    signal['reason'] = f'Negative sentiment ({sentiment_score:.2f}) with continued decline ({price_change:.2f}%) - Exit position!'
                else:
                    signal['action'] = 'SELL'
                    signal['reason'] = f'Negative sentiment ({sentiment_score:.2f}) with stable price - Risk management!'
        
        return signal
    
    def process_news_and_generate_signals(self):
        """Main method to process news and generate signals"""
        print("🔄 Processing news and generating trading signals...")
        news_items = self.fetch_crypto_news()
        
        if not news_items:
            print("❌ No news items found")
            return
        
        print(f"📰 Processing {len(news_items)} news items")
        
        # Process each news item
        for news in news_items:
            title = news.get('title', '')
            description = news.get('description', '')
            content = news.get('content', '')
            
            full_text = f"{title}. {description}. {content[:500]}"
            
            # Analyze sentiment
            sentiment_data = self.analyze_sentiment(full_text)
            
            # Check which assets are mentioned
            mentioned_assets = []
            if self.demo_mode and 'assets' in news:
                mentioned_assets = news['assets']
            else:
                for symbol, name in self.assets.items():
                    if (symbol.lower() in full_text.lower() or 
                        name.lower() in full_text.lower() or
                        symbol in title.upper() or
                        name in title):
                        mentioned_assets.append(symbol)
            
            # Generate signals for mentioned assets
            for asset in mentioned_assets:
                price_data = self.get_price_data(asset)
                signal = self.generate_trading_signal(asset, sentiment_data, price_data)
                
                if signal and signal['action'] != 'HOLD':
                    self.trading_signals.append(signal)
                    print(f"🎯 {signal['action']} signal for {asset}: {signal['reason']}")
                
                # Store sentiment data
                self.sentiment_scores.append({
                    'asset': asset,
                    'timestamp': datetime.now().isoformat(),
                    'sentiment': sentiment_data['sentiment'],
                    'confidence': sentiment_data['confidence'],
                    'news_title': title[:100] + "..." if len(title) > 100 else title
                })
    
    def get_dashboard_data(self):
        """Get data for dashboard display"""
        return {
            'signals': self.trading_signals[-10:],
            'sentiments': self.sentiment_scores[-20:],
            'assets': self.assets,
            'last_update': datetime.now().isoformat(),
            'demo_mode': self.demo_mode
        }

# Initialize the demo agent
demo_agent = DemoNewsTradingAgent(demo_mode=True)

# Flask app for web interface
app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    return jsonify(demo_agent.get_dashboard_data())

@app.route('/api/process')
def process_news():
    demo_agent.process_news_and_generate_signals()
    return jsonify({'status': 'success', 'message': 'News processed successfully'})

@app.route('/api/demo')
def demo_mode():
    return jsonify({
        'status': 'success', 
        'message': 'Demo mode active - guaranteed trading signals!',
        'demo_mode': True
    })

def run_demo_loop():
    """Run the demo agent in a continuous loop"""
    while True:
        try:
            demo_agent.process_news_and_generate_signals()
            print(f"⏰ Demo mode: Next update in 2 minutes...")
            time.sleep(120)  # Wait 2 minutes for demo
        except KeyboardInterrupt:
            print("🛑 Demo agent stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in demo loop: {e}")
            time.sleep(30)

if __name__ == "__main__":
    print("🎭 Starting AI News Trading Agent - DEMO MODE")
    print("📊 Supported assets:", list(demo_agent.assets.keys()))
    print("🌐 Web dashboard available at: http://localhost:5001")
    print("🎯 Demo mode: Guaranteed trading signals for presentation!")
    
    # Start the demo agent in a separate thread
    agent_thread = threading.Thread(target=run_demo_loop, daemon=True)
    agent_thread.start()
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5001) 