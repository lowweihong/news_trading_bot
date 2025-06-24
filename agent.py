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

# Load environment variables from .env
load_dotenv()
API_KEY = os.getenv("SUPRA_API_KEY")
REST_BASE_URL = "https://prod-kline-rest.supra.com"
HISTORY_ENDPOINT = "/history"
LATEST_ENDPOINT = "/latest"

class NewsTradingAgent:
    def __init__(self):
        self.news_queue = queue.Queue()
        self.price_data = {}
        self.trading_signals = []
        self.sentiment_scores = []
        
        # Trading parameters
        self.sentiment_threshold = 0.3  # Positive sentiment threshold
        self.price_change_threshold = 0.02  # 2% price change threshold
        self.confidence_threshold = 0.6  # Minimum confidence for signals
        
        # Supported assets
        self.assets = {
            'btc_usd': 'Bitcoin',
            'eth_usd': 'Ethereum', 
            'supra_usd': 'Supra',
        }
        
    def fetch_crypto_news(self):
        """Fetch crypto news from BlockBeat API"""
        try:
            # Using BlockBeat API for news
            url = "https://api.theblockbeats.news/v1/open-api/open-information"
            params = {
                'size': 10,
                'page': 1,
                'lang': 'en'
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                news_data = response.json()
                if news_data.get('status') == 0:  # BlockBeat uses status 0 for success
                    return news_data.get('data', {}).get('data', [])
                else:
                    print(f"BlockBeat API error: {news_data.get('message', 'Unknown error')}")
                    return []
            else:
                print(f"Failed to fetch news: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of news text using TextBlob"""
        try:
            blob = TextBlob(text)
            # Get polarity (-1 to 1, where -1 is very negative, 1 is very positive)
            sentiment_score = blob.sentiment.polarity
            # Get subjectivity (0 to 1, where 0 is objective, 1 is subjective)
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
        """Get real-time price data from Supra price feeds"""
        try:
            # Using Supra API for price data
            url = f"{REST_BASE_URL}{LATEST_ENDPOINT}"
            headers = {"x-api-key": API_KEY}
            params = {"trading_pair": symbol}  # Use symbol directly (e.g., btc_usd)
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Supra API response for {symbol}: {data}")  # Debug output
                
                # Handle different possible response formats
                if isinstance(data, dict):
                    # Try to extract price data from various possible structures
                    current_price = (
                        data.get('price') or 
                        data.get('close') or 
                        data.get('last_price') or 
                        data.get('data', {}).get('price') or 
                        0
                    )
                    price_change = (
                        data.get('price_change_24h') or 
                        data.get('change_24h') or 
                        data.get('data', {}).get('change_24h') or 
                        0
                    )
                    
                    return {
                        'symbol': symbol,
                        'price': float(current_price) if current_price else 0,
                        'price_change_24h': float(price_change) if price_change else 0,
                        'volume': data.get('volume', 0),
                        'market_cap': data.get('market_cap', 0),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    print(f"Unexpected Supra API response format for {symbol}")
                    return self._get_yfinance_fallback(symbol)
            else:
                print(f"Supra API error for {symbol}: {response.status_code}")
                # Fallback to yfinance if Supra fails
                return self._get_yfinance_fallback(symbol)
                
        except Exception as e:
            print(f"Error fetching price data for {symbol}: {e}")
            # Fallback to yfinance
            return self._get_yfinance_fallback(symbol)
    
    def _get_yfinance_fallback(self, symbol):
        """Fallback method using yfinance if Supra API fails"""
        try:
            # Convert btc_usd to BTC-USD format for yfinance
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
        """Generate trading signal based on sentiment and price data"""
        if not price_data:
            return None
            
        sentiment_score = sentiment_data['sentiment']
        confidence = sentiment_data['confidence']
        price_change = price_data['price_change_24h']
        
        # Signal logic
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
        
        # Buy signal conditions
        if (sentiment_score > self.sentiment_threshold and 
            confidence > self.confidence_threshold and
            abs(price_change) < 10):  # Avoid extreme volatility
            
            if price_change < -self.price_change_threshold:  # Price dropped, sentiment positive
                signal['action'] = 'BUY'
                signal['reason'] = f'Positive sentiment ({sentiment_score:.2f}) with recent price drop ({price_change:.2f}%)'
            elif price_change > self.price_change_threshold:  # Price up, strong sentiment
                signal['action'] = 'BUY'
                signal['reason'] = f'Strong positive sentiment ({sentiment_score:.2f}) with upward momentum ({price_change:.2f}%)'
        
        # Sell signal conditions
        elif (sentiment_score < -self.sentiment_threshold and 
              confidence > self.confidence_threshold):
            
            if price_change > self.price_change_threshold:  # Price up, negative sentiment
                signal['action'] = 'SELL'
                signal['reason'] = f'Negative sentiment ({sentiment_score:.2f}) despite price increase ({price_change:.2f}%)'
            elif price_change < -self.price_change_threshold:  # Price down, negative sentiment
                signal['action'] = 'SELL'
                signal['reason'] = f'Negative sentiment ({sentiment_score:.2f}) with continued decline ({price_change:.2f}%)'
        
        return signal
    
    def process_news_and_generate_signals(self):
        """Main method to process news and generate trading signals"""
        print("🔄 Fetching latest crypto news from BlockBeat...")
        news_items = self.fetch_crypto_news()
        
        if not news_items:
            print("❌ No news items found")
            return
        
        print(f"📰 Found {len(news_items)} news items from BlockBeat")
        
        # Process each news item
        for news in news_items[:10]:  # Limit to 10 most recent
            title = news.get('title', '')
            description = news.get('description', '')
            content = news.get('content', '')
            
            # Combine title, description, and content for sentiment analysis
            full_text = f"{title}. {description}. {content[:500]}"  # Limit content length
            
            # Analyze sentiment
            sentiment_data = self.analyze_sentiment(full_text)
            
            # Check which assets are mentioned
            mentioned_assets = []
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
            'signals': self.trading_signals[-10:],  # Last 10 signals
            'sentiments': self.sentiment_scores[-20:],  # Last 20 sentiment scores
            'assets': self.assets,
            'last_update': datetime.now().isoformat()
        }

# Initialize the agent
agent = NewsTradingAgent()

# Flask app for web interface
app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    return jsonify(agent.get_dashboard_data())

@app.route('/api/process')
def process_news():
    agent.process_news_and_generate_signals()
    return jsonify({'status': 'success', 'message': 'News processed successfully'})

def run_agent_loop():
    """Run the agent in a continuous loop"""
    while True:
        try:
            agent.process_news_and_generate_signals()
            print(f"⏰ Next update in 5 minutes...")
            time.sleep(300)  # Wait 5 minutes
        except KeyboardInterrupt:
            print("🛑 Agent stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in agent loop: {e}")
            time.sleep(60)  # Wait 1 minute on error

if __name__ == "__main__":
    print("🚀 Starting AI News Trading Agent...")
    print("📊 Supported assets:", list(agent.assets.keys()))
    print("🌐 Web dashboard available at: http://localhost:5001")
    
    # Start the agent in a separate thread
    agent_thread = threading.Thread(target=run_agent_loop, daemon=True)
    agent_thread.start()
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)
