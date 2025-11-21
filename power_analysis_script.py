# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from ee_sample_data import generate_power_data

# Load and Explore Power Data
def load_and_explore_data():
    data = generate_power_data()
    print("First few rows of the dataset:")
    print(data.head())
    print("\nMissing values:")
    print(data.isnull().sum())
    print("\nBasic statistics:")
    print(data.describe())
    return data

# Analyze Power Consumption Trends
def analyze_power_consumption(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    daily_consumption = data['power_consumption'].resample('D').sum()
    plt.figure(figsize=(10, 6))
    plt.plot(daily_consumption, label='Daily Power Consumption')
    plt.title('Daily Power Consumption Trends')
    plt.xlabel('Date')
    plt.ylabel('Power Consumption (kWh)')
    plt.legend()
    plt.show()

# Analyze Power Quality Metrics
def analyze_power_quality(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data[['voltage', 'frequency']])
    plt.title('Power Quality Metrics')
    plt.show()

# Main Function
def main():
    data = load_and_explore_data()
    analyze_power_consumption(data)
    analyze_power_quality(data)

if __name__ == "__main__":
    main()