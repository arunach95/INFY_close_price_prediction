# Stock Closing Price Prediction Using News Sentiment Analysis and Market Data

This project predicts the **closing price of Infosys (INFY.NS)** by integrating **historical stock data** with **news sentiment analysis**. It utilizes **machine learning**, **web scraping**, and **natural language processing** to enhance stock market predictions.

## 📌 Features

- **Web Scraping**: Extracts **Infosys-related news** using **Selenium** and **stock data** from NSE.
- **Sentiment Analysis**: Uses **ProsusAI/finbert** to analyze financial news sentiment.
- **Stock Price Prediction**: Implements a **Random Forest Regressor**, optimized using **GridSearchCV**.
- **Deployment**: A **Streamlit web app** allows real-time stock price prediction at the click of a button.

st.markdown("![Alt Text](https://example.com/image.jpg)")

## 🚀 Future Enhancements

- Expand the model to predict prices for **multiple stocks**.
- Improve sentiment analysis by analyzing **full news articles** instead of just headlines.
- Replace **web scraping** with **financial APIs** for faster, more reliable data collection.
- Enhance **UI/UX** by upgrading from Streamlit to a more advanced framework.

## 🛠️ Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/arunach95/INFY_close_price_prediction
   cd UI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run streamlit-app.py
   ```

## 📜 License

This project is for educational purposes. Feel free to explore and contribute!

---

🔗 **Author**: *Aruna C H* 
