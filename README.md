# Understanding Algorithmic Sentiment Analysis on Financial News Headlines

The code employs Natural Language Processing (NLP) techniques to analyze sentiment in financial news headlines, aiming to gain insights into market sentiment for specific stocks. We will walk through the code step by step, providing explanations and examples to aid comprehension.

## Importing Libraries and Downloading NLTK Data
The initial lines of code import necessary libraries, such as Pandas for data manipulation, urllib for web requests, NLTK for Natural Language Processing, and specifically, the VADER sentiment analysis tool. The code also downloads the VADER lexicon, a pre-trained sentiment analysis dictionary used by the NLTK library.

```
import pandas as pd
from urllib.request import Request, urlopen
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime

nltk.download('vader_lexicon')
```

## Defining Parameters
The script then defines parameters crucial for subsequent operations. num_headlines specifies the number of article headlines to be displayed per ticker, and tickers is a list of stock symbols (e.g., AAPL, TSLA, AMZN) to be analyzed.
```
num_headlines = 3
tickers = ['AAPL', 'TSLA', 'AMZN']
```

## Initializing SentimentIntensityAnalyzer
The SentimentIntensityAnalyzer from NLTK is initialized to perform sentiment analysis on the news headlines.
```
analyzer = SentimentIntensityAnalyzer()
```
## Fetching and Processing News Data
The fetch_news function is defined to retrieve and process news data for a given stock symbol. It uses the Finviz website to scrape recent headlines for the specified ticker. The function returns a Pandas DataFrame containing the datetime and headline for the requested number of headlines.
```
def fetch_news(ticker):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'Mozilla/5.0'})
    response = urlopen(req)
    df = pd.read_html(response.read(), attrs={'id': 'news-table'})[0]
    df.columns = ['Datetime', 'Headline']
    return df.head(num_headlines)
```
## Performing Sentiment Analysis
The sentiment_analysis function is designed to analyze sentiment for each headline in a given DataFrame. It utilizes the VADER sentiment analysis tool, which assigns a compound sentiment score to each headline.
```
def sentiment_analysis(df):
    df['Sentiment'] = df['Headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    return df
```
## Processing and Analyzing News for Each Ticker
The code iterates through the list of tickers, fetching and processing news data for each one. It then prints the recent news headlines and their corresponding sentiment scores.
```
for ticker in tickers:
    news_df = fetch_news(ticker)
    news_df['Ticker'] = ticker
    sentiment_df = sentiment_analysis(news_df)
    print('\nRecent News Headlines and Sentiment for {}: '.format(ticker))
    print(sentiment_df)
```
## Calculating Mean Sentiment for Each Ticker
The script calculates the mean sentiment for each ticker by grouping the news DataFrame by ticker and computing the mean sentiment score.
```
mean_sentiments = {ticker: sentiment_analysis(fetch_news(ticker))['Sentiment'].mean() for ticker in tickers}
```
## Creating and Displaying DataFrame of Mean Sentiment Scores
Finally, the code constructs a Pandas DataFrame containing the mean sentiment scores for each ticker and prints the result, providing an overview of the sentiment trends for the selected stocks.
```
sentiments_df = pd.DataFrame(list(mean_sentiments.items()), columns=['Ticker', 'Mean Sentiment'])
sentiments_df = sentiments_df.set_index('Ticker').sort_values('Mean Sentiment', ascending=False)
print(sentiments_df)
```
In summary, this code provides a framework for sentiment analysis on financial news headlines for specific stocks. By leveraging NLP and the VADER sentiment analysis tool, it aims to quantify the sentiment associated with recent news, offering valuable insights for algorithmic trading strategies.
