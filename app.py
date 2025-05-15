import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
import warnings

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warning
warnings.filterwarnings('ignore')
plt.style.use("fivethirtyeight")

# Initialize Flask app
app = Flask(__name__)

# Create static directory if it doesn't exist
os.makedirs('static', exist_ok=True)

# Custom LSTM class to handle unsupported args like 'time_major'
class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Remove unrecognized argument
        super().__init__(*args, **kwargs)

# Load model with custom object
try:
    model = load_model('stock_dl_model.h5', custom_objects={'LSTM': CustomLSTM}, compile=False)
except Exception as e:
    raise RuntimeError(f"Model loading failed. Make sure 'stock_dl_model.h5' exists. Error: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock', 'POWERGRID.NS')  # Default value
        
        try:
            # Define date range
            start = dt.datetime(2000, 1, 1)
            end = dt.datetime(2024, 10, 1)

            # Download stock data
            df = yf.download(stock, start=start, end=end)
            if df.empty:
                return "No data found for this stock symbol.", 400

            # Calculate EMAs
            emas = {
                'ema20': df['Close'].ewm(span=20, adjust=False).mean(),
                'ema50': df['Close'].ewm(span=50, adjust=False).mean(),
                'ema100': df['Close'].ewm(span=100, adjust=False).mean(),
                'ema200': df['Close'].ewm(span=200, adjust=False).mean()
            }

            # Data processing
            split_idx = int(len(df) * 0.70)
            data_training = pd.DataFrame(df['Close'][:split_idx])
            data_testing = pd.DataFrame(df['Close'][split_idx:])

            # Scaling
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)

            # Prepare prediction data
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.transform(final_df)

            # Create sequences
            x_test, y_test = [], []
            for i in range(100, len(input_data)):
                x_test.append(input_data[i - 100:i])
                y_test.append(data_testing.iloc[i - 100])  # Use original unscaled close values

            x_test, y_test = np.array(x_test), np.array(y_test)

            # Make predictions
            y_predicted = model.predict(x_test)

            # Inverse scaling
            scale_factor = 1 / scaler.scale_[0]
            y_predicted = y_predicted.flatten() * scale_factor
            y_test = y_test * scale_factor  # Scale back y_test for plotting

            # Function to save plots
            def save_plot(data1, data2, data3, title, filename):
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data1.values, 'y', label='Closing Price')
                
                # Safely get name or use fallback
                label2 = getattr(data2, 'name', 'Line 2')
                label3 = getattr(data3, 'name', 'Line 3')

                ax.plot(data2.values, 'g', label=label2)
                ax.plot(data3.values, 'r', label=label3)
                ax.set_title(title)
                ax.set_xlabel("Time")
                ax.set_ylabel("Price")
                ax.legend()
                path = f"static/{filename}"
                fig.savefig(path)
                plt.close(fig)
                return path

            # Save EMA charts
            ema_chart_path = save_plot(df.Close, emas['ema20'], emas['ema50'],
                                       "Closing Price vs Time (20 & 50 Days EMA)",
                                       "ema_20_50.png")

            ema_chart_path_100_200 = save_plot(df.Close, emas['ema100'], emas['ema200'],
                                              "Closing Price vs Time (100 & 200 Days EMA)",
                                              "ema_100_200.png")

            # Prediction plot
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(y_test, 'g', label="Original Price", linewidth=1)
            ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
            ax3.set_title("Prediction vs Original Trend")
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Price")
            ax3.legend()
            prediction_chart_path = "static/stock_prediction.png"
            fig3.savefig(prediction_chart_path)
            plt.close(fig3)

            # Save dataset
            csv_file_path = f"static/{stock}_dataset.csv"
            df.to_csv(csv_file_path)

            # Return rendered template
            return render_template('index.html',
                                   plot_path_ema_20_50='ema_20_50.png',
                                   plot_path_ema_100_200='ema_100_200.png',
                                   plot_path_prediction='stock_prediction.png',
                                   data_desc=df.describe().to_html(classes='table table-bordered'),
                                   dataset_link=f"{stock}_dataset.csv")

        except Exception as e:
            return f"Error: {str(e)}", 500

    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(f"static/{filename}", as_attachment=True, download_name=filename)
    except FileNotFoundError:
        return "File not found", 404


if __name__ == '__main__':
    app.run(debug=True)