from flask import Flask, render_template, jsonify
import joblib
import pandas as pd
import yfinance as yf

app = Flask(__name__)

# Load model
model = joblib.load("doge_price_movement_model.pkl")

# Feature columns
feature_columns = ['Close', 'Daily Return', 'MA7', 'MA20', 'MA50', 'Volatility', 'Volume']


def get_latest_live_row():
    ticker = yf.Ticker("DOGE-USD")
    df_live = ticker.history(period="1d").reset_index()
    return df_live.iloc[-1]


def prepare_live_features(dataset, live_row):
    temp = dataset[['Close', 'Volume']].copy()
    temp = temp.reset_index(drop=True)

    temp.loc[len(temp)] = [live_row['Close'], live_row['Volume']]

    temp['Daily Return'] = temp['Close'].pct_change()
    temp['MA7'] = temp['Close'].rolling(7).mean()
    temp['MA20'] = temp['Close'].rolling(20).mean()
    temp['MA50'] = temp['Close'].rolling(50).mean()
    temp['Volatility'] = temp['Daily Return'].rolling(7).std()

    return temp.tail(1)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict_live():
    try:
        # Fetch historical price for chart (last 7 days)
        hist = yf.Ticker("DOGE-USD").history(period="7d")
        hist = hist.reset_index()

        live_row = hist.iloc[-1]  # latest candle

        # Load main dataset
        dataset = pd.read_csv("DOGE-USD (3).csv")

        # Prepare features for prediction
        features = prepare_live_features(dataset, live_row)
        X = features[feature_columns]

        pred = model.predict(X)[0]

        if pred == 1:
            result = "📈 Dogecoin is likely to GO UP tomorrow"
        else:
            result = "📉 Dogecoin may GO DOWN tomorrow"

        # Prepare graph data
        dates = hist['Date'].dt.strftime("%Y-%m-%d").tolist()
        prices = hist['Close'].tolist()

        # Live price
        current_price = live_row['Close']

        return jsonify({
            "prediction": int(pred),
            "message": result,
            "current_price": float(current_price),
            "history_dates": dates,
            "history_prices": prices
        })

    except Exception as e:
        return jsonify({"error": str(e)})

  

if __name__ == "__main__":
    app.run(debug=True)
