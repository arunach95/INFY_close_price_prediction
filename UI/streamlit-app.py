import streamlit as st
import time
import pandas as pd
import pickle
import numpy as np
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from webdriver_manager.chrome import ChromeDriverManager

# ---------- Helper Functions ----------

def setup_driver():
    """Set up the Selenium WebDriver."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver

def scrape_google_news(driver):
    """Scrape Google News for Infosys-related articles."""
    driver.get("https://news.google.com/search?q=infosys%20when%3A1d&hl=en-IN&gl=IN&ceid=IN%3Aen")
    time.sleep(5)
    try:
        articles = driver.find_elements(By.CLASS_NAME, "JtKRv")
        articles_text = [article.text for article in articles]
    except Exception as e:
        st.error(f"Error scraping Google News: {e}")
        articles_text = []
    finally:
        driver.quit()
    return articles_text

def analyze_sentiment(articles_text):
    """Perform sentiment analysis on the articles."""
    MODEL_NAME = "ProsusAI/finbert"
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    results = sentiment_analyzer(articles_text)
    labels = [result["label"] for result in results]
    scores = [result["score"] for result in results]
    return labels, scores

def calculate_sentiment_mode(labels, scores):
    """Calculate the mode of the sentiment and its mean score."""
    labels_series = pd.Series(labels)
    sentiment_mode = labels_series.mode()[0]
    df = pd.DataFrame({'Sentiment': labels, 'Sentiment Score': scores})
    sentiment_mode_data = df[df['Sentiment'] == sentiment_mode]
    mean_sentiment_score = sentiment_mode_data['Sentiment Score'].mean()
    return sentiment_mode, mean_sentiment_score

def scrape_nse_data(driver):
    """Scrape NSE website for stock data."""
    driver.get("https://www.nseindia.com/get-quotes/equity?symbol=INFY")
    time.sleep(5)
    data = {}
    try:
        ltp_element = driver.find_element(By.ID, "quoteLtp")
        data["Last Traded Price (LTP)"] = ltp_element.text
        table_xpath = '//table[@id="priceInfoTable"]/tbody/tr/td'
        table_cells = driver.find_elements(By.XPATH, table_xpath)
        table_headers = ["Prev. Close", "Open", "High", "Low", "Close", "Indicative Close", "VWAP", "Adjusted Price"]
        table_values = [cell.text for cell in table_cells]
        for header, value in zip(table_headers, table_values):
            data[header] = value
    except Exception as e:
        st.error(f"Error scraping NSE data: {e}")
    finally:
        driver.quit()
    return data

def preprocess_features(data, sentiment_mode, mean_sentiment_score):
    """Preprocess features for prediction."""
    sentiment_positive = sentiment_mode == 'positive'
    sentiment_negative = sentiment_mode == 'negative'
    sentiment_neutral = sentiment_mode == 'neutral'

    features = pd.DataFrame([{
        "OPEN": data['Open'],
        "HIGH": data['High'],
        "LOW": data['Low'],
        "PREV. CLOSE": data['Prev. Close'],
        "ltp": data['Last Traded Price (LTP)'],
        "vwap": data['VWAP'],
        "Sentiment Score": mean_sentiment_score,
        "Sentiment_negative": sentiment_negative,
        "Sentiment_neutral": sentiment_neutral,
        "Sentiment_positive": sentiment_positive
    }])

    columns_to_convert = ["OPEN", "HIGH", "LOW", "PREV. CLOSE", "ltp", "vwap"]
    for column in columns_to_convert:
        features[column] = features[column].astype(str)
        features[column] = features[column].str.replace(",", "", regex=False)  # Remove commas
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features

def load_model():
    """Load the saved stock price prediction model."""
    with open("stock_price_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def predict_price(model, features):
    """Predict the stock price using the model."""
    predicted_price = model.predict(features)
    return predicted_price[0]

# ---------- Streamlit App ----------

def main():
    # Initialize session states
    if "popup_acknowledged" not in st.session_state:
        st.session_state.popup_acknowledged = False
    if "predict_clicked" not in st.session_state:
        st.session_state.predict_clicked = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard"

    # Show welcome popup if not acknowledged yet
    if not st.session_state.popup_acknowledged:
        # Create a full-screen popup with about content
        st.title("📊 Welcome to Infosys Stock Price Prediction App")
        
        # Add a decorative line
        st.markdown("<hr style='margin: 15px 0px; border: 2px solid #FF4B4B;'>", unsafe_allow_html=True)
        
        st.markdown("""
        ### 🔍 About This Application
        
        This application was developed as part of a **final-year academic project** to explore the use of **machine learning** and **real-time data analysis** in stock price prediction.  
        It integrates **financial market trends** and **sentiment analysis** to provide meaningful insights into potential stock movements.  

        ### ⚙️ How It Works:
        - **Live news data** is analyzed to assess public sentiment and predict its impact on stock prices.  
        - **Live stock market data** is used to capture key fundamentals like opening price, volume, and trends.  
        - **Machine learning models** process this data to generate experimental stock price predictions.  

        ### ⚠️ Important Disclaimer:
        This application is purely for **academic and experimental purposes**.  
        Predictions made by the model are not guaranteed to be accurate—**machine learning models can make mistakes**.  
        This is **not a stock recommendation tool**. Investors should conduct their own research and consult a financial expert before making any investment decisions.  

        ### 👨‍💻 Project Details:
        - **Developer**: Aruna C H  (2382410)
        - **Program**: MSc Data Analytics 
        - **Institution**: CHRIST (Deemed to be University)  
        - **Trimester**: VI (January 2025)  
        """)
        
        # Center the accept button using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ I Understand and Accept", use_container_width=True):
                st.session_state.popup_acknowledged = True
                st.rerun()  # Rerun the app to show the main content
                
    else:
        # Regular app content after popup is acknowledged
        # Sidebar Navigation
        st.sidebar.image("UI/infosys.png", use_container_width=True)
        st.sidebar.title("Quick Navigation")
        page = st.sidebar.radio("Discover insights, track trends, and explore predictions to stay ahead in the market.", ["🏠 Dashboard", "ℹ️ About the App"])

        # Reset session state when navigating between pages
        if st.session_state.current_page != page:
            st.session_state.current_page = page
            st.session_state.predict_clicked = False  # Reset the button state

        if page == "🏠 Dashboard":
            # Dashboard Page
            st.title("📈 Infosys Stock Price Prediction")
            st.markdown("""
            Welcome to the **Infosys Stock Price Prediction App**!  
            Harnessing the power of **machine learning and real-time data**, this application delivers intelligent forecasts for Infosys' closing stock price.
            Our advanced prediction model leverages:
            - 📊 **Live NSE market data** to capture market trends  
            - 📰 **Sentiment analysis of financial news** to gauge investor sentiment  
            - 🤖 **Machine Learning Algorithms** for accurate predictions
            """)

            # Add a section for user interaction
            st.header("🔮 Predict Stock Price")

            if not st.session_state.predict_clicked:
                st.write("Click the button below to predict the stock price:")
                if st.button("Predict Stock Price"):
                    st.session_state.predict_clicked = True  # Set the flag to True
                    st.rerun()  # Force rerun to update UI immediately
            else:
                st.write("⏳ Hold tight! The magic is happening...")

                # Step 1: Scrape Google News for Infosys-related articles
                with st.spinner("Scraping Google News..."):
                    driver = setup_driver()
                    articles_text = scrape_google_news(driver)
                if not articles_text:
                    st.error("No articles scraped. Please try again later.")
                    return
                st.success("✅ News articles scraped successfully!")

                # Step 2: Perform sentiment analysis on the articles
                with st.spinner("Analyzing sentiment..."):
                    labels, scores = analyze_sentiment(articles_text)
                    sentiment_mode, mean_sentiment_score = calculate_sentiment_mode(labels, scores)
                st.write(f"**Sentiment Mode:** {sentiment_mode}")
                st.write(f"**Mean Sentiment Score:** {mean_sentiment_score:.2f}")

                # Step 3: Scrape NSE data for Infosys
                with st.spinner("Scraping NSE data..."):
                    driver = setup_driver()  # Reinitialize driver for NSE scraping
                    nse_data = scrape_nse_data(driver)
                st.success("✅ NSE data scraped successfully!")

                # Step 4: Preprocess features
                with st.spinner("Preprocessing features..."):
                    features = preprocess_features(nse_data, sentiment_mode, mean_sentiment_score)
                st.success("✅ Features preprocessed successfully!")

                # Step 5: Load model and predict stock price
                with st.spinner("Loading model and predicting price..."):
                    model = load_model()
                    predicted_price = predict_price(model, features)
                st.success(f"🎉 Predicted Closing Price: **₹{predicted_price:.2f}**")

                # Disclaimer
                st.markdown("""
                **Disclaimer:**  
                This prediction is for **experimental and educational purposes only**.  
                Stock market movements are influenced by various unpredictable factors, and **machine learning models can make errors**.  
                This is **not financial advice**. Please conduct your own research and consult a financial expert before making any investment decisions.  
                
                ---
                Developed by [Aruna C H](https://github.com/arunach95/INFY_close_price_prediction.git)                    
                """)


        elif page == "ℹ️ About the App":
            # About the App Page - same content as the initial popup
            st.title("About the Infosys Stock Price Prediction App")
            st.markdown("""
            This application was developed as part of a **final-year academic project** to explore the use of **machine learning** and **real-time data analysis** in stock price prediction.  
            It integrates **financial market trends** and **sentiment analysis** to provide meaningful insights into potential stock movements.  

            ## How It Works:
            - **Live news data** is analyzed to assess public sentiment and predict its impact on stock prices.  
            - **Live stock market data** is used to capture key fundamentals like opening price, volume, and trends.  
            - **Machine learning models** process this data to generate experimental stock price predictions.  

            ## Important Disclaimer:
            This application is purely for **academic and experimental purposes**.  
            Predictions made by the model are not guaranteed to be accurate—**machine learning models can make mistakes**.  
            This is **not a stock recommendation tool**. Investors should conduct their own research and consult financial experts before making any investment decisions.  

            ## Project Details:
            - **Developer**: Aruna C H  (2382410)
            - **Program**: MSc Data Analytics 
            - **Institution**: CHRIST (Deemed to be University)  
            - **Trimester**: VI (January 2025)  

            ## Acknowledgments:
            This project was made possible with the support of faculty and mentors.  
            It leverages **Python, Streamlit, Selenium, Transformers, and Scikit-learn** for data analysis and modeling.

            ---
            Developed by [Aruna C H](https://github.com/arunach95/INFY_close_price_prediction.git)
            """)

if __name__ == "__main__":
    main()