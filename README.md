# stock_price_prediction
This project predicts future stock prices using an LSTM (Long Short-Term Memory) neural network trained on historical data. It features an interactive Streamlit web app for visualizing real vs predicted prices,
## 🧠 Project Overview

Stock price prediction is a classic time series problem in finance. LSTM networks are especially suited for sequential data as they can capture **long-term dependencies**. This project:

- Downloads historical stock data using `yfinance`
- Visualizes closing prices and moving averages
- Predicts future stock prices using a pre-trained LSTM model
- Evaluates model performance using metrics like **RMSE, MAE, R² score, and approximate accuracy**
- Provides an interactive interface via **Streamlit** for end-users

---

## ⚙️ Features

- Input a stock ticker symbol and view historical stock data
- Visualize closing prices, 100-day MA, and 200-day MA
- Predict future stock prices with LSTM
- Compare original vs predicted prices in interactive plots
- Model evaluation metrics displayed to assess accuracy

---

## 📂 Project Structure

stock_price_prediction/
│
├── main.py ← Main Streamlit app
├── keras_model.h5 ← Pre-trained LSTM model
├── requirements.txt ← Python dependencies



## 🧩 Tech Stack

- **Frontend/UI:** Streamlit  
- **Backend/Model:** TensorFlow, Keras  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Matplotlib  
- **Stock Data:** yfinance  
- **Deployment:** Streamlit Cloud  

---

## 🧮 Model Details

The model is a **stacked LSTM network** trained on historical closing prices:

- **Input:** Past 100 days of closing prices  
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  
- **Scaling:** MinMaxScaler (0-1)  



```bash
git clone https://github.com/<your-username>/stock_price_prediction.git
cd stock_price_prediction
