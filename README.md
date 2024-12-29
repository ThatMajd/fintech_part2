# **Project Title**

> This project focuses on forecasting the daily price of Gold Futures by gathering historical market data via the yfinance module. After exploring potential drivers of gold prices—such as related commodities, indices, and currency exchange rates—the data is fed into both classic time series models and machine learning algorithms through PyCaret. By comparing various predictive approaches, the project aims to identify which model best captures the trends and seasonality in Gold Futures, ultimately offering reliable short-term price forecasts.

---

## **Table of Contents**

1. [Data Collection](#data-collection)
2. [Data Visualizing and Cleaning](#data-visualizing-and-cleaning)
3. [Models](#models)
    - [Time Series Models](#time-series-models)
    - [Machine Learning Models](#machine-learning-models)
    - [Models Comparison](#models-comparison)
4. [Trading Strategy](#trading-strategy)
5. [Setting Up the Environment](#environment-setup)


## **Data Collection**

Data collection involved extracting historical daily price data for Gold Futures, along with selected commodities and market indices, using the yfinance Python library. This approach provided a convenient way to gather accurate, up-to-date financial information for a specified timeframe. By consolidating price data from multiple sources, the project was able to explore a range of potential market drivers influencing Gold Futures, forming a solid foundation for subsequent exploratory analysis and model development. The data was stored in [Data Directory](./data/).

---

## **Data Visualizing and Cleaning**
We used custom Python scripts and modules to pull historical price data for Gold Futures (and related market assets) via yfinance. We then performed initial preprocessing steps—such as addressing missing values, standardizing date formats, and removing duplicate entries—to ensure the integrity and consistency of the dataset with multiple plots in [Notebooks](./data_scraping_cleaning/) before proceeding with deeper exploratory analysis and modeling.

## **Models**
### **Time Series Models**
Before we started to run Time Series models, we made sure of the data by running [Time Series Analysis](./Time%20Series%20Analysis/), after that, We implemented up to 21 classic time series models, including ARIMA, SARIMA, and Exponential Smoothing and many more, to capture the temporal dependencies and seasonality in the Gold Futures data using Pycaret. These models were evaluated based on their forecasting accuracy and ability to handle different patterns in the historical data. The [Output](optimized_model_comparison_results.csv) tells that the most promising was the Auto-arima model which resembles the SARIMA model, so we fine-tuned the model and performed [Grid Search](sarima_grid_search.py) to catch the best combination of parameters, the final best performing SARIMA can be found in [Models Directory](./models/) and results can be found in [Plots](./plots/). We also implemented Prophet of Meta and stores the results in [Plots](./plots/).

### **Machine Learning Models**
We explored various machine learning algorithms such as Random Forest, Gradient Boosting, and Support Vector Machines. These models were trained on the historical price data and evaluated using cross-validation techniques to ensure robust performance. The goal was to identify the model that provides the most accurate and reliable short-term forecasts for Gold Futures. We concluded that the best performing was the LSTM model, you find the implementation under [Models Directory](./models/) and results in [Plots](./plots/)

### **Models Comparison**
Each model was trained on 80%, and predicted on 20% od the data. We used these prediction as our main indicator of model performance and to choose whether or not we captured trends and movements of the price in a good way. Comparisons can be found under this [directory](./Time%20Series%20Analysis/).


## **Trading Strategy**
Lastly, we made a moderate performing [trading strategy](./Trading%20Strategy/trading.ipynb) based on predictions, we used the Close Price of the Gold Futures to decide when to buy and sell, and used additional parameters that could affect the decisions.

## **Environment Setup**
1. **Ensure you have Python installed**:
   - You can download Python from [python.org](https://www.python.org/).
2. **Install the required libraries**:
   - Run the following command in your terminal to install all the dependencies listed in the `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```
    You can look up the file clicking [here](./requirement.txt)

Thanks for checking out our work! :) 



