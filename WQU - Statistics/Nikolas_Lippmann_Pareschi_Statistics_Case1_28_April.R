# First of all, we need to download the Nasdaq data from yahoo finance

nasdaq <- read.csv(file = "http://chart.finance.yahoo.com/table.csv?s=^IXIC&a=3&b=27&c=2014&d=3&e=27&f=2017&g=d&ignore=.csv", header = TRUE, stringsAsFactors = FALSE)
nasdaq
cot_nasdaq = nasdaq[, "Adj.Close", drop = FALSE]

# We need to reverse the Nasdaq cotations. If we do not do that, the daily returns will be negative as we will be "going back to the past"

cot_nasdaq2 <- apply(cot_nasdaq, 2, rev)
cot_nasdaq2

# We have sucessfully reversed and we have saved the number of rows in the dataframe do n

n <- nrow(cot_nasdaq2)

# We need to calculate the daily returns. First we will calculate for the entire history. 

nasdaq_daily_returns <- ((cot_nasdaq2[2:n, 1] - cot_nasdaq2[1:(n-1), 1]) / cot_nasdaq2[1:(n-1), 1])

# Ploting a histogram

hist(nasdaq_daily_returns, main = "Nasdaq Daily Returns - All history", breaks = 40)

# Are the returns normally distributed? We will use the Shapiro-Wilk test to check that. 

shapiro.test(nasdaq_daily_returns)

# As we can see, the normal distribution hypothesis is rejected for in the 95% confidence level (the 95% level is the benchmark of that test)
# Let´s calculate the mean, the standard deviation and de median for our daily returns, considering all historic.


mx = mean(nasdaq_daily_returns)
mx
sd = sd(nasdaq_daily_returns)
sd
medx = median(nasdaq_daily_returns)
medx
hist(nasdaq_daily_returns, main = "Nasdaq Daily Returns - All history", breaks = 40)

#add a line for the mean

abline(v = mean(nasdaq_daily_returns),
       col = "blue",
       lwd = 2)

#add a line for the median

abline(v = median(nasdaq_daily_returns),
       col = "red",
       lwd = 2)

#add a line for the standard deviation

abline(v = sd(nasdaq_daily_returns), col = "green", lwd = 2)

#add legends

legend(x = "topright",
       c("Mean", "Median","Std Dev"),
       col = c("blue", "red", "green"),
       lwd = c(2,2,2))



# Now let's do the same calculations, but for just 1 year, as asked!

nasdaq_daily_returns2 <- ((cot_nasdaq2[2:253, 1] - cot_nasdaq2[1:252, 1]) / cot_nasdaq2[1:252, 1])
hist(nasdaq_daily_returns2, main = "Nasdaq Daily Returns - 1 year", breaks = 40)
mx2 = mean(nasdaq_daily_returns2)
mx2
sd2 = sd(nasdaq_daily_returns2)
sd2
medx2 = median(nasdaq_daily_returns2)
medx2
hist(nasdaq_daily_returns2, main = "Nasdaq Daily Returns - 1 year", breaks = 40)

# Let´s perform the Shapiro-Walk test for the normal hypothesis.

shapiro.test(nasdaq_daily_returns2)

#add a line for the mean

abline(v = mean(nasdaq_daily_returns),
       col = "blue",
       lwd = 2)

#add a line for the median

abline(v = median(nasdaq_daily_returns),
       col = "red",
       lwd = 2)

#add a line for the standard deviation

abline(v = sd(nasdaq_daily_returns), col = "green", lwd = 2)

#adding legends

legend(x = "topright",
       c("Mean", "Median","Std Dev"),
       col = c("blue", "red", "green"),
       lwd = c(2,2,2))


# As you will see we need to reverse the dataframe

plot(nasdaq$Adj.Close, type = "l", col = "blue", lwd = 2, 
     ylab = "Adjusted close", 
     main = "Daily close prices of Nasdaq, need to reverse!")

legend(x = 'topleft', legend = 'NASDAQ', lty = 1, lwd = 2, col = 'blue')


# Reversing the dataframe

nasdaq2 <- apply(nasdaq, 2, rev)
nasdaq2

plot(nasdaq2[,7], type = "l", col = "blue", lwd = 2, 
     ylab = "Adjusted close", 
     main = "Daily close prices of NASDAQ")

legend(x = 'topleft', legend = 'NASDAQ', lty = 1, lwd = 2, col = 'blue')
nasdaq_daily_returns2