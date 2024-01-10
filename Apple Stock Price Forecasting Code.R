#Business Forecasting Project - R code
# 80% Train - Fit model 
# 20% Test; Forecast values and use accuracy function to compute errors, model which will give least error is best model
# Project AIM : TO find best model for our data set and forecast and predict future values
#Choose a Time Series Data Set....
library(fpp2)
library(readxl)

# Importing Apple dataset in projectData variable:
projectData <- read_excel("~/Desktop/Business Forecasting/BF Project/aapl.xlsx",sheet = "AAPL.Close_Updated")

#Extracting close price in appl_close variable:
appl_close <- projectData$AAPL.Close

#Making close price of Apple(appl_close) into time-series data with frequency 253:
# frequency is 253 as in one year we have only 253 data entries(records)
#2018 has 232 records, year 2007 till 2017 has 253 records 
appl_close_ts <- ts(data=appl_close, frequency=253, start=c(2007,1))

#Plotting close price to observe the behaviour of the data:
autoplot(appl_close_ts) + xlab("Year") + ylab("Closing Price of Apple")+
  ggtitle("Timeseries plot of Apple closing price")

# Autocorrelation to find the trend and seasonality:
ggAcf(appl_close_ts)
#Constant decrease in  ACF is due to trend.

#Lag plots:
gglagplot(appl_close_ts) 

# Splitting data in Traning and Testing(80:20) set for forecasting:
# 3015 total records, 2412 training and 603 is test set so we will use h=603
appl_training <- window(appl_close_ts, start = c(2007,1), end=c(2016,135))
appl_testing <- window(appl_close_ts, start= c(2016,136), end=c(2018,232))

# Plotting Traning and testing set:
autoplot(appl_training,series="Train") +
  autolayer(appl_testing,series="Test") +
  xlab("Year") + ylab("Closing Price of Apple stock ") +
  ggtitle("Training and Testing Time series dataset") +
  guides(colour=guide_legend(title="AppleStock"))


#data is our apple_close ts
#if we remove trend and seasonal cycle from original data the remaining is known as remainder)
#Decomposing apple close price of stock to analyze Trend- Cycle and Seasonality using STL:
#s.window is the number of consecutive years to be used in estimating 
#each value in the seasonal component. The user must specify s.window as there
#is no default. Setting it to be infinite is equivalent to forcing 
#the seasonal component to be periodic (i.e., identical across years).
appl_stl<-stl(appl_close_ts, s.window = 5)

#STL decomposed components plot
plot(appl_stl)

#Data plotted with Trend component
plot(appl_close_ts, col="gray",
     main="Apple Stock Price",
     ylab="Apple Stock price in $", xlab="Time")

lines(appl_stl$time.series[,"trend"],col="red",ylab="Trend")

#Seasonal adjusted plot
plot(appl_close_ts, col="grey",
     main="Apple Stock Price",
     ylab="Apple Stock Price in $", xlab="Time")

seasadj(appl_stl)
lines(seasadj(appl_stl),col="green",ylab="Seasonally adjusted")


# Train Mean model for Closing Price: 
appl_mean <- meanf(appl_training,h=603)

#Forecast of Apple Closing Price using Mean method:
autoplot(appl_mean) +
  xlab("Year") + ylab("Closing Price of Apple stock") +
  ggtitle("Forecasts from Mean method")

#Check Residuals
checkresiduals(appl_mean)

# Compute forecast accuracy measures of mean method: 
accuracy(appl_mean, appl_testing)

# Train Naive model for Closing Price:
appl_naive <- rwf(appl_training,h=603)

#Forecast of Apple Closing Price using Naive method:
autoplot(appl_naive) +
  xlab("Year") + ylab("Closing Price of Apple stock") +
  ggtitle("Forecasts from Naive method")

#Check Residuals
checkresiduals(appl_naive)

# Compute forecast accuracy measures of Naive method: 
#RMSE of training set is 1.1457 and of test set 68.436
accuracy(appl_naive, appl_testing)

# Train Seasonal Naive model for Closing Price: 
appl_snaive <- snaive(appl_training,h=603)

#Forecast of Apple Closing Price using Seasonal Naive method:
autoplot(appl_snaive) +
  xlab("Year") + ylab("Closing Price of Apple stock") +
  ggtitle("Forecasts from Seasonal Naive method")

#Check Residuals
checkresiduals(appl_snaive)

# Compute forecast accuracy measures of Seasonal Naive method: 
accuracy(appl_snaive, appl_testing)

# Train Drift model for Closing Price:
appl_drift <- rwf(appl_training,h=603,drift=TRUE)

#Forecast of Apple Closing Price using Drift method:
autoplot(appl_drift) +
  xlab("Year") + ylab("Closing Price of Apple stock") +
  ggtitle("Forecasts from Drift method")

#Check Residuals
checkresiduals(appl_drift)

# Compute forecast accuracy measures of Drift method: 
# RMSE of training set is 1.145233 and Test set is 56.0401
accuracy(appl_drift, appl_testing)


# Train regression model on Apple closing price:
# Running regression model with Trend and season as the predictor variables 
appl_reg <- tslm(appl_training ~ trend + season)

# Forecast of Apple Closing Price using Regression Model:
appl_reg_forecast <- forecast(appl_reg, h= 603)

# Plot of forecast to test data for regression:
autoplot(appl_reg_forecast) +
  autolayer(appl_testing, series = "Test") +
  xlab("Year") + ylab("Closing Price of Apple stock") +
  ggtitle("Forecasts of Apple stock closing price using regression") +
  guides(colour = guide_legend(title = "Data"))

#Check Residuals
checkresiduals(appl_reg)

# Compute forecast accuracy measures of Regression method:
accuracy(appl_reg_forecast, appl_testing)

# Forecast of Apple Closing Price using Holt's linear trend method:

appl_holt <- holt(appl_training, h= 603)


# Smoothing Parameters:
#AIC = 19450.20
#BIC = 19479.15 
appl_holt[["model"]]

# or use appl_holt$model

# Plot of forecast to test data for Holt's linear:
autoplot(appl_holt, series = "Forecast") +
  autolayer(appl_testing, series = "Test") +
  xlab("Year") + ylab("Closing Price of Apple stock") +
  ggtitle("Forecasts of Apple stock closing price using holt") +
  guides(colour = guide_legend(title = "Data"))

#Check Residuals
checkresiduals(appl_holt)


# Compute forecast accuracy measures of Holt's linear:
# RMSE of training set is 1.145438, test set is 57.298729 
accuracy (appl_holt, appl_testing)


#ARIMA Model
#Using Auto.Arima to fit best ARIMA model(finding p,d,q)
fit <- auto.arima(appl_training)
fit
#AIC=7500.02, BIC = 7511.6
# ARIMA(0,1,0) with drift
# sigma^2 estimated as 1.312:  log likelihood=-3748.01

# Fitting ARIMA(0,1,0) for apple training data:
fit2 <- Arima(appl_training, order=c(0,1,0),include.drift = TRUE)
appl_arima <- forecast(fit2, h=603)

# Plot of forecast to test data for Arima:
autoplot(appl_arima, series = "Forecast") +
  autolayer(appl_testing, series = "Test") +
  xlab("Year") + ylab("Closing Price of Apple stock") +
  ggtitle("Forecasts of Apple stock closing price using Arima(0,1,0)") +
  guides(colour = guide_legend(title = "Data"))

#Check Residuals
checkresiduals(appl_arima)

# Compute forecast accuracy measures of ARIMA(0,1,0) model with drift:
# RMSE for training set is 1.144996 and test set is 56.040182
accuracy (appl_arima, appl_testing)


# After comparing all the models we found out that drift method 
# is the best with good accuracy on test data:
# Now we will forecast future values: 
# For that we will train model on the whole dataset:
future_drift <- rwf(appl_close_ts,h= 253, drift = TRUE)

autoplot(future_drift) +
  xlab("Year") + ylab("Closing Price of Apple stock") +
  ggtitle("Forecasts from Drift method")

# Forecast values:
forecast(future_drift)
