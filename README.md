# AI News Trading Agent 🚀

An intelligent trading agent that combines **BlockBeat news sentiment analysis** with **Supra price feeds** to generate real-time buy/sell signals for cryptocurrency trading.

## 🎯 Features

- **Real-time News Analysis**: Fetches latest crypto news from BlockBeat API
- **Sentiment Analysis**: Uses TextBlob for AI-powered sentiment scoring
- **Price Data Integration**: Connects to Supra price feeds for real-time market data
- **Smart Signal Generation**: Combines sentiment + price action for trading decisions
- **Beautiful Dashboard**: Modern web interface with real-time updates
- **Multiple Asset Support**: BTC, ETH, SOL, ADA, DOT and more

## 🏆 Perfect for Supra Hackathon

This project demonstrates:
- **AI Agents + Supra Integration**: Seamless connection between AI sentiment analysis and Supra's on-chain price feeds
- **Real-time Decision Making**: Intelligent trading signals based on market conditions
- **Hybrid On-chain/Off-chain Setup**: Combines off-chain news sentiment with on-chain price data
- **Production Ready**: Complete with web dashboard and API endpoints

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
The Supra API key is already configured in the code. For production use, consider using environment variables.

### 3. Run the Agent
```bash
python full_agent.py
```

### 4. Access Dashboard
Open your browser and go to: `http://localhost:5002`

## 📊 How It Works

### 1. News Collection
- Fetches latest crypto news from BlockBeat API
- Processes title, description, and content for comprehensive analysis

### 2. Sentiment Analysis
- Uses TextBlob for natural language processing
- Calculates sentiment polarity (-1 to +1) and confidence scores
- Identifies mentioned cryptocurrencies in news content

### 3. Price Data Integration
- Connects to Supra price feeds for real-time market data
- Falls back to yfinance if Supra API is unavailable
- Tracks price changes, volume, and market cap

### 4. Signal Generation
**Buy Signals:**
- Positive sentiment (>0.3) + high confidence (>0.6)
- Price drop with positive news (buy the dip)
- Strong upward momentum with positive sentiment

**Sell Signals:**
- Negative sentiment (<-0.3) + high confidence
- Price increase despite negative news
- Continued decline with negative sentiment

### 5. Real-time Dashboard
- Live signal updates
- Sentiment trend visualization
- Asset distribution charts
- Historical signal tracking

## 🛠️ API Endpoints

- `GET /` - Main dashboard
- `GET /api/data` - Get current trading data
- `POST /api/process` - Manually trigger news processing

## 📈 Trading Strategy

The agent uses a sophisticated approach combining:

1. **Sentiment Threshold**: 0.3 (positive/negative)
2. **Confidence Threshold**: 0.6 (minimum signal confidence)
3. **Price Change Threshold**: 2% (significant price movement)
4. **Volatility Protection**: Avoids signals during extreme volatility (>10% price swings)

## 🔧 Configuration

You can adjust trading parameters in the `NewsTradingAgent` class:

```python
self.sentiment_threshold = 0.3      # Sentiment sensitivity
self.price_change_threshold = 0.02  # Price movement threshold
self.confidence_threshold = 0.6     # Minimum confidence for signals
```

## 📱 Demo Assets: ETH

Easily extendable by adding more assets to the `assets` dictionary.

## 🎨 Dashboard Features

- **Real-time Updates**: Auto-refreshes every 30 seconds
- **Signal Visualization**: Color-coded buy/sell/hold signals
- **Sentiment Charts**: Interactive sentiment trend analysis
- **Asset Distribution**: Doughnut chart showing signal distribution
- **Historical Data**: Track sentiment and signal history
- **Responsive Design**: Works on desktop and mobile

## 🔒 Security & Best Practices

- API key management (use environment variables in production)
- Rate limiting for API calls
- Error handling and fallback mechanisms
- Input validation and sanitization

## 🚨 Disclaimer

This is a **demo project** for educational purposes. Do not use for actual trading without:
- Thorough backtesting
- Risk management implementation
- Professional financial advice
- Understanding of market risks

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License - feel free to use this project for your Supra hackathon submission!

---

**Built with ❤️ for the Supra AI Agents Hackathon** 