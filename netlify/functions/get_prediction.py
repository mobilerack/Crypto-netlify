import json
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression

def get_crypto_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {'vs_currency': 'usd', 'days': 90, 'interval': 'daily'}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json().get('prices', [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        return df[['date', 'price']]
    except requests.exceptions.RequestException as e:
        print(f"Hiba az API hívás során: {e}")
        return pd.DataFrame()

def train_and_predict(df):
    if df is None or len(df) < 2:
        return 0
    df['target'] = df['price'].shift(-1)
    df.dropna(inplace=True)
    X = df[['price']]
    y = df['target']
    model = LinearRegression()
    model.fit(X, y)
    last_known_price = df[['price']].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(last_known_price)
    return prediction[0]

def handler(event, context):
    crypto_df = get_crypto_data()
    predicted_price = train_and_predict(crypto_df.copy())
    labels = [date.strftime('%Y-%m-%d') for date in crypto_df['date']]
    prices = [price for price in crypto_df['price']]
    response_data = {
        'prediction': predicted_price,
        'labels': labels,
        'prices': prices
    }
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps(response_data)
    }
