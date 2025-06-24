#!/usr/bin/env python3
"""
Simple test script for the AI News Trading Agent
"""

from agent import NewsTradingAgent
import json

def test_agent():
    print("🧪 Testing AI News Trading Agent...")
    
    # Initialize agent
    agent = NewsTradingAgent()
    
    # Test news fetching
    print("\n📰 Testing news fetching from BlockBeat...")
    news_items = agent.fetch_crypto_news()
    print(f"Found {len(news_items)} news items")
    
    if news_items:
        print("\n📋 Sample news item:")
        sample_news = news_items[0]
        print(f"Title: {sample_news.get('title', 'N/A')[:100]}...")
        print(f"Description: {sample_news.get('description', 'N/A')[:100]}...")
        
        # Test sentiment analysis
        print("\n🧠 Testing sentiment analysis...")
        title = sample_news.get('title', '')
        description = sample_news.get('description', '')
        content = sample_news.get('content', '')
        full_text = f"{title}. {description}. {content[:500]}"
        
        sentiment_data = agent.analyze_sentiment(full_text)
        print(f"Sentiment Score: {sentiment_data['sentiment']:.3f}")
        print(f"Confidence: {sentiment_data['confidence']:.3f}")
        print(f"Subjectivity: {sentiment_data['subjectivity']:.3f}")
        
        # Test price data fetching
        print("\n💰 Testing price data fetching...")
        for asset in ['BTC', 'ETH']:
            price_data = agent.get_price_data(asset)
            if price_data:
                print(f"{asset}: ${price_data['price']:.2f} ({price_data['price_change_24h']:.2f}%)")
            else:
                print(f"{asset}: Failed to fetch price data")
        
        # Test signal generation
        print("\n🎯 Testing signal generation...")
        agent.process_news_and_generate_signals()
        
        # Show results
        print(f"\n📊 Generated {len(agent.trading_signals)} trading signals")
        for signal in agent.trading_signals:
            print(f"  {signal['asset']} - {signal['action']}: {signal['reason']}")
        
        print(f"\n📈 Collected {len(agent.sentiment_scores)} sentiment scores")
        
        # Show dashboard data
        print("\n📋 Dashboard data:")
        dashboard_data = agent.get_dashboard_data()
        print(json.dumps(dashboard_data, indent=2, default=str))
        
    else:
        print("❌ No news items found. Check API connectivity.")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_agent() 