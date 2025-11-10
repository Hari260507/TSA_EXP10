# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL 
## DATE: 17/102025
### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the Tomato dataset
data = pd.read_csv("Tomato.csv")

# If your file has a date column, use it; otherwise, create one
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
else:
    # Create a daily date range for demonstration (adjust frequency if needed)
    data['date'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')

# Set the date column as index
data.set_index('date', inplace=True)

# Plot the Average price over time
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Average'])
plt.xlabel('Date')
plt.ylabel('Average Tomato Price')
plt.title('Tomato Average Price Time Series')
plt.show()

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Check stationarity
check_stationarity(data['Average'])

# Plot ACF and PACF
plot_acf(data['Average'])
plt.show()

plot_pacf(data['Average'])
plt.show()

# Train-test split
train_size = int(len(data) * 0.8)
train, test = data['Average'][:train_size], data['Average'][train_size:]

# Fit SARIMA model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Forecast
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1)

# Evaluate model
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('Root Mean Squared Error (RMSE):', rmse)

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Average Tomato Price')
plt.title('SARIMA Model Predictions for Tomato Prices')
plt.legend()
plt.show()



```
### OUTPUT:
<img width="1090" height="725" alt="image" src="https://github.com/user-attachments/assets/9c3a826c-d750-44af-945d-61ea9d410923" />
<img width="712" height="552" alt="image" src="https://github.com/user-attachments/assets/bfb70d75-6d48-441e-9d79-bac356eb542b" />
<img width="1279" height="655" alt="image" src="https://github.com/user-attachments/assets/27525ad1-91fe-4580-bb71-674432283f25" />
<img width="1039" height="582" alt="image" src="https://github.com/user-attachments/assets/7258a58d-af91-487e-bf4b-11811ecbeb1c" />



### RESULT:
Thus the program run successfully based on the SARIMA model.
