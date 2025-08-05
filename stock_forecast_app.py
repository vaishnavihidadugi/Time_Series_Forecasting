import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import tempfile

# 📌 Helper Function: Train & Forecast with Model
def train_model(df, model_name):
    data = df.filter(['Close'])
    dataset = data.values
    train_size = int(len(dataset) * 0.8)
    train, test = dataset[:train_size], dataset[train_size:]
    rmse, forecast = None, None

    if model_name == "ARIMA":
        model = ARIMA(train, order=(5, 1, 0)).fit()
        forecast = model.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test, forecast))

    elif model_name == "Prophet":
        prophet_df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        prophet_train = prophet_df[:train_size]
        prophet_model = Prophet(daily_seasonality=True).fit(prophet_train)
        future = prophet_model.make_future_dataframe(periods=len(test))
        forecast_df = prophet_model.predict(future)
        forecast = forecast_df['yhat'].values[-len(test):]
        rmse = np.sqrt(mean_squared_error(test, forecast))

    elif model_name == "LSTM":
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        x_train, y_train = [], []
        for i in range(60, len(train)):
            x_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            LSTM(50, return_sequences=False),
            Dense(25), Dense(1)
        ])
        lstm.compile(optimizer='adam', loss='mean_squared_error')
        lstm.fit(x_train, y_train, batch_size=1, epochs=1)
        x_test = []
        test_scaled = scaled_data[train_size - 60:, :]
        for i in range(60, len(test_scaled)):
            x_test.append(test_scaled[i-60:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predictions = lstm.predict(x_test)
        forecast = scaler.inverse_transform(predictions).flatten()
        rmse = np.sqrt(mean_squared_error(test, forecast))

    return forecast, rmse

# 📌 Main Forecast Function
def forecast_stock(csv_file, selection_mode, selected_model):
    try:
        df = pd.read_csv(csv_file.name)
        if 'Date' not in df.columns or 'Close' not in df.columns:
            return None, "❌ CSV must contain 'Date' and 'Close' columns.", None, None

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        train_size = int(len(df) * 0.8)
        test = df['Close'][train_size:]

        rmse_table = []
        best_model, forecast, rmse_value = selected_model, None, None

        if selection_mode == "Auto":
            models = ["ARIMA", "Prophet", "LSTM"]
            results = {}
            for m in models:
                f, r = train_model(df, m)
                results[m] = {"forecast": f, "rmse": r}
                rmse_table.append({"Model": m, "RMSE": round(r, 3)})
            best_model = min(results, key=lambda x: results[x]['rmse'])
            forecast = results[best_model]['forecast']
            rmse_value = results[best_model]['rmse']
        else:
            forecast, rmse_value = train_model(df, selected_model)
            rmse_table = [{"Model": selected_model, "RMSE": round(rmse_value, 3)}]

        # 📊 Plot
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 6))
        plt.plot(test.index, test.values, label='Actual', color='lime')
        plt.plot(test.index, forecast, label='Predicted', color='deepskyblue')
        plt.title(f"{best_model} Forecast vs Actual", fontsize=16, color='white')
        plt.xlabel('Date', color='white')
        plt.ylabel('Close Price', color='white')
        plt.legend()
        plt.grid(True, color='gray')
        plot_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plt.savefig(plot_file.name, facecolor='black')
        plt.close()

        # 📥 Save predictions
        pred_df = pd.DataFrame({'Date': test.index, 'Predicted': forecast})
        pred_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        pred_df.to_csv(pred_file.name, index=False)

        rmse_df = pd.DataFrame(rmse_table)
        return plot_file.name, f"✅ {best_model} selected! RMSE: {rmse_value:.3f}", pred_file.name, rmse_df

    except Exception as e:
        return None, f"❌ Error: {str(e)}", None, None

# 🌌 Gradio Dark Neon Theme UI
# 🌌 Gradio Dark Neon Theme UI
css = """
body { background-color: #121212; color: #E0E0E0; font-family: 'Segoe UI', sans-serif; }
h1, h2, h3 { color: #00c6ff; }
.gr-button { background: linear-gradient(45deg, #00c6ff, #0072ff); color: white; border-radius: 10px; }
.gr-button:hover { background: linear-gradient(45deg, #ff6ec4, #7873f5); }
.gr-file, .gr-dropdown, .gr-radio, .gr-textbox, .gr-dataframe {
  border: 1px solid #333333;
  background-color: #1E1E1E;
  color: #E0E0E0;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## 🌌 Stock Price Forecasting App")
    gr.Markdown("Upload a stock CSV (`Date` & `Close` columns). Auto lets AI pick the best model, Manual lets you choose.")

    csv_input = gr.File(label="📁 Upload Stock CSV")
    selection_mode = gr.Radio(["Auto", "Manual"], value="Auto", label="🔀 Mode")
    model_dropdown = gr.Dropdown(["ARIMA", "Prophet", "LSTM"], label="📊 Choose Model (if Manual)", visible=False)

    plot_output = gr.Image(label="📈 Forecast Plot")
    status_output = gr.Textbox(label="📢 Status")
    download_output = gr.File(label="📥 Download Forecast CSV")
    rmse_table_output = gr.Dataframe(label="📋 RMSE Comparison Table")

    def toggle_model_dropdown(mode): return gr.update(visible=(mode == "Manual"))
    selection_mode.change(toggle_model_dropdown, selection_mode, model_dropdown)

    run_btn = gr.Button("🚀 Run Forecast")
    run_btn.click(forecast_stock,
                  inputs=[csv_input, selection_mode, model_dropdown],
                  outputs=[plot_output, status_output, download_output, rmse_table_output])

demo.launch()
