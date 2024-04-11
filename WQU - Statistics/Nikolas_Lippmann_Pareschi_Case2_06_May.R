
###### 1 -Download data for last 1 years for a set of the any five stock tickers belonging to the same industry segment ########

# To accomplish what was asked we first create the function below.
# It is important to notice that the function inverts the tables provided by yahoo with the historical prices.
# The tables from yahoo have the first data as the most new. If we do not invert, our plot would be inverted.

#DISCLOSRURE: I used this function but I am not the author, I found it in github.


download_data = function(url) {
  raw = read.table(url, header=TRUE, sep=",")
  raw = raw[, c(1, 7)] # 1 is Date column, 7 is Adj.Close column
  raw = raw[nrow(raw):1, ] # Sort oldest to newest
  return(raw)
}

# We then select 5 companies in the sector of healthcare plans"

aet_url = "http://chart.finance.yahoo.com/table.csv?s=AET&a=4&b=1&c=2016&d=5&e=1&f=2017&g=m&ignore=.csv"
antm_url = "http://chart.finance.yahoo.com/table.csv?s=ANTM&a=4&b=1&c=2016&d=5&e=1&f=2017&g=m&ignore=.csv"
esrx_url = "http://chart.finance.yahoo.com/table.csv?s=ESRX&a=4&b=1&c=2016&d=5&e=1&f=2017&g=m&ignore=.csv"
ci_url= "http://chart.finance.yahoo.com/table.csv?s=CI&a=4&b=1&c=2016&d=5&e=1&f=2017&g=m&ignore=.csv"
unh_url = "http://chart.finance.yahoo.com/table.csv?s=UNH&a=4&b=1&c=2016&d=5&e=1&f=2017&g=m&ignore=.csv"

# We download the data

#Aetna In

AET = download_data(aet_url)
AET

#Anthem, Inc.

ANTM = download_data(antm_url)
ANTM

#Express Scripts Holdng Company"

ESRX = download_data(esrx_url)
ESRX

#Cigna Corporation

CI= download_data(ci_url)
CI

#United Health Group Incorporated"

UNH = download_data(unh_url)
UNH

############# 2. Calculate Monthly returns of downloaded stock over the period under study ###################

# We need a function to calculate monthly returns of downloaded stock over the period under study
# Disclosure. The author of this function is Supasate Choochaisri. It is in github



monthly_returns = function(data) {
  close_values = data[, 2] # 2 is Adj.Close column
  num = length(close_values) - 1
  returns = numeric(num)
  for (i in 1:num) {
    returns[i] = (close_values[i + 1] - close_values[i])/close_values[i]
  }
  return(returns)
}

# Monthly returns, cumulative returns for AET, ANTM, ESRX, CI, UNH - Healthcare companies

# AET


AET_returns = monthly_returns(AET)
AET_returns
sd(AET_returns)
AET_returns2 = AET_returns + 1
AET_returns2
AETcumprod = cumprod(AET_returns2)


#ANTM

ANTM_returns = monthly_returns(ANTM)
ANTM_returns
sd(ANTM_returns)
ANTM_returns2 = ANTM_returns + 1
ANTMcumprod = cumprod(ANTM_returns2)


#ESRX

ESRX_returns = monthly_returns(ESRX)
ESRX_returns
sd(ESRX_returns)
ESRX_returns2 = ESRX_returns + 1
ESRXcumprod = cumprod(ESRX_returns2)


#CI

CI_returns = monthly_returns(CI)
CI_returns
sd(CI_returns)
CI_returns2 = CI_returns + 1
CIcumprod = cumprod(CI_returns2)


#UNH

UNH_returns = monthly_returns(UNH)
UNH_returns
sd(UNH_returns)
UNH_returns2 = UNH_returns + 1
UNHcumprod = cumprod(UNH_returns2)


# Hraphs for the cumulative return of each company studied in the healhplan sector

plot(AETcumprod, type = "l", col = "blue", lwd = 2, ylab = "Cumulative return", main = "Cumulative returns of AET")
plot(ANTMcumprod, type = "l", col = "red", lwd = 2, ylab = "Cumulative return", main = "Cumulative returns of ANTM")
plot(ESRXcumprod, type = "l", col = "green", lwd = 2, ylab = "Cumulative return", main = "Cumulative returns of ERSX")
plot(CIcumprod, type = "l", col = "yellow", lwd = 2, ylab = "Cumulative return", main = "Cumulative returns of CI")
plot(UNHcumprod, type = "l", col = "black", lwd = 2, ylab = "Cumulative return", main = "Cumulative returns of UNH")


########## Using a combination function, calculate the monthly returns of an equally weighted portfolio consisting of any 3 of the five stocks in question######

#First we create a dataframe of all companies and its respectives cumulative returns

stock_retcumulative <- data.frame(AETcumprod, ANTMcumprod, ESRXcumprod, CIcumprod, UNHcumprod)
stock_retcumulative

#Then we apply the combn function

combin <- combn(stock_retcumulative, 3, simplify = FALSE)
combin 

# Now we pass the combn results for the 10 portfolios and we also calculate the mean, median and standard deviation which is
# another step asked

############ 5. Calculate mean, median and standard deviation of monthly values for each of the portfolios in question
#and plot them on the same graph mentioned in step 4.###############



portfolio1 <- combin[[1]]
portfolio1["Portfolio Return"] <- (portfolio1[1]+portfolio1[2]+portfolio1[3])/3
portfolio1
portfolio1_mean <- (mean(ESRX_returns) + mean(AET_returns) + mean(ANTM_returns))/3
portfolio1_mean
portfolio1_median <- median((ESRX_returns + AET_returns + ANTM_returns)/3)
portfolio1_median
portfolio1_sd <- sd((ESRX_returns + AET_returns + ANTM_returns)/3)
portfolio1_sd
portfolio1_sharp <- (portfolio1_mean/portfolio1_sd)
portfolio1_sharp

portfolio2 <- combin[[2]]
portfolio2["Portfolio Return"] <- (portfolio2[1]+portfolio2[2]+portfolio2[3])/3
portfolio2
portfolio2_mean <- (mean(ANTM_returns) + mean(AET_returns) + mean(CI_returns))/3
portfolio2_mean
portfolio2_median <- median((CI_returns + AET_returns + ANTM_returns)/3)
portfolio2_median
portfolio2_sd <- sd((CI_returns + AET_returns + ANTM_returns)/3)
portfolio2_sd
portfolio2_sharp <- (portfolio2_mean/portfolio2_sd)
portfolio2_sharp

portfolio3 <- combin[[3]]
portfolio3["Portfolio Return"] <- (portfolio3[1]+portfolio3[2]+portfolio3[3])/3
portfolio3
portfolio3_mean <- (mean(UNH_returns) + mean(AET_returns) + mean(ANTM_returns))/3
portfolio3_mean
portfolio3_median <- median((UNH_returns + AET_returns + ANTM_returns)/3)
portfolio3_median
portfolio3_sd <- sd((UNH_returns + AET_returns + ANTM_returns)/3)
portfolio3_sd
portfolio3_sharp <- (portfolio3_mean/portfolio3_sd)
portfolio3_sharp

portfolio4 <- combin[[4]]
portfolio4["Portfolio Return"] <- (portfolio4[1]+portfolio4[2]+portfolio4[3])/3
portfolio4
portfolio4_mean <- (mean(ESRX_returns) + mean(AET_returns) + mean(CI_returns))/3
portfolio4_mean
portfolio4_median <- median((ESRX_returns + AET_returns + CI_returns)/3)
portfolio4_median
portfolio4_sd <- sd((ESRX_returns + AET_returns + CI_returns)/3)
portfolio4_sd
portfolio4_sharp <- (portfolio4_mean/portfolio4_sd)
portfolio4_sharp

portfolio5 <- combin[[5]]
portfolio5["Portfolio Return"] <- (portfolio5[1]+portfolio5[2]+portfolio5[3])/3
portfolio5
portfolio5_mean <- (mean(ESRX_returns) + mean(AET_returns) + mean(UNH_returns))/3
portfolio5_mean
portfolio5_median <- median((ESRX_returns + AET_returns + UNH_returns)/3)
portfolio5_median
portfolio5_sd <- sd((ESRX_returns + AET_returns + UNH_returns)/3)
portfolio5_sd
portfolio5_sharp <- (portfolio5_mean/portfolio5_sd)
portfolio5_sharp

portfolio6 <- combin[[6]]
portfolio6["Portfolio Return"] <- (portfolio6[1]+portfolio6[2]+portfolio6[3])/3
portfolio6
portfolio6_mean <- (mean(UNH_returns) + mean(AET_returns) + mean(CI_returns))/3
portfolio6_mean
portfolio6_median <- median((UNH_returns + AET_returns + CI_returns)/3)
portfolio6_median
portfolio6_sd <- sd((UNH_returns + AET_returns + CI_returns)/3)
portfolio6_sd
portfolio6_sharp <- (portfolio6_mean/portfolio6_sd)
portfolio6_sharp

portfolio7 <- combin[[7]]
portfolio7["Portfolio Return"] <- (portfolio7[1]+portfolio7[2]+portfolio7[3])/3
portfolio7
portfolio7_mean <- (mean(ANTM_returns) + mean(ESRX_returns) + mean(CI_returns))/3
portfolio7_mean
portfolio7_median <- median((ESRX_returns + ANTM_returns + CI_returns)/3)
portfolio7_median
portfolio7_sd <- sd((ESRX_returns + ANTM_returns + CI_returns)/3)
portfolio7_sd
portfolio7_sharp <- (portfolio7_mean/portfolio7_sd)
portfolio7_sharp

portfolio8 <- combin[[8]]
portfolio8["Portfolio Return"] <- (portfolio8[1]+portfolio8[2]+portfolio8[3])/3
portfolio8
portfolio8_mean <- (mean(ANTM_returns) + mean(ESRX_returns) + mean(UNH_returns))/3
portfolio8_mean
portfolio8_median <- median((ESRX_returns + ANTM_returns + UNH_returns)/3)
portfolio8_median
portfolio8_sd <- sd((ESRX_returns + ANTM_returns + UNH_returns)/3)
portfolio8_sd
portfolio8_sharp <- (portfolio8_mean/portfolio8_sd)
portfolio8_sharp

portfolio9 <- combin[[9]]
portfolio9["Portfolio Return"] <- (portfolio9[1]+portfolio9[2]+portfolio9[3])/3
portfolio9
portfolio9_mean <- (mean(ANTM_returns) + mean(CI_returns) + mean(UNH_returns))/3
portfolio9_mean
portfolio9_median <- median((CI_returns + ANTM_returns + UNH_returns)/3)
portfolio9_median
portfolio9_sd <- sd((CI_returns + ANTM_returns + UNH_returns)/3)
portfolio9_sd
portfolio9_sharp <- (portfolio9_mean/portfolio9_sd)
portfolio9_sharp

portfolio10 <- combin[[10]]
portfolio10["Portfolio Return"] <- (portfolio10[1]+portfolio10[2]+portfolio10[3])/3
portfolio10
portfolio10_mean <- (mean(ESRX_returns) + mean(CI_returns) + mean(UNH_returns))/3
portfolio10_mean
portfolio10_median <- median((CI_returns + ESRX_returns + UNH_returns)/3)
portfolio10_median
portfolio10_sd <- sd((CI_returns + ESRX_returns + UNH_returns)/3)
portfolio10_sd
portfolio10_sharp <- (portfolio10_mean/portfolio10_sd)
portfolio10_sharp

portfolio1["Portfolio Return"]

########

SRcomparison <- c(portfolio1_sharp, portfolio2_sharp, portfolio3_sharp, portfolio4_sharp, portfolio5_sharp, portfolio6_sharp, portfolio7_sharp, portfolio8_sharp, portfolio9_sharp, portfolio10_sharp)
SRcomparison

######## Graphically represent the cumulative monthly returns of each of the possible portfolios through line plots ###########



plot(portfolio1["Portfolio Return"], type = "l", col = "red", main = "All 10 Portfolios", xlab = "Month", ylab = "Return 1 to 1.45", xlim=c(1, 12), ylim=c(0.9, 1.45))
lines(portfolio1["Portfolio Return"], col = "purple")
lines(portfolio2["Portfolio Return"], col = "blue")
lines(portfolio3["Portfolio Return"], col = "pink")
lines(portfolio4["Portfolio Return"], col = "royalblue")
lines(portfolio5["Portfolio Return"], col = "orange")
lines(portfolio6["Portfolio Return"], col = "black")
lines(portfolio7["Portfolio Return"], col = "magenta")
lines(portfolio8["Portfolio Return"], col = "yellow")
lines(portfolio9["Portfolio Return"], col = "grey")
lines(portfolio10["Portfolio Return"], col = "green")



#Adding legends

legend("topleft", legend = c("P1, mean = 0.0118 median = -0.001 sd = 0.0594", "P2, mean = 0.0237 median = 0.0142 sd = 0.0597", "P3, mean = 0.0251 median = 0.0116 sd = 0.0558", "P4, mean = 0.0089 median = 0.0046 sd = 0.0579", "P5, mean = 0.0104 median = -0.0062 sd = 0.0546", "P6, mean = 0.0223 median = 0.0163 sd = 0.0536", "P7, mean = 0.0113 median = 0.0020 sd = 0.0521", "P8, mean = 0.0127 median = 0.0051 sd = 0.0475", "P9, mean = 0.0246 median = 0.0152 sd = 0.0489", "P10, mean = 0.0098 median = 0.0082 sd = 0.0455"),
       col=c("purple", "blue","pink", "royalblue", "orange", "black", "magenta", "yellow","grey","green" ), lty=1:2, cex=0.8,
       text.font=.1)


############## 6. Calculate the overall variance of all portfolio returns ###############

#As the portfolios are not independent of each other, we cannot just sum the variances and then divide by the number of portfolios.
# Instead, we can use a trick. The variance to all porfolios combined will be the variance of a portfolio of equal weights to all the five shares.
# So lets calculate that.

totalreturn <- (AET_returns + CI_returns + ESRX_returns + UNH_returns + ANTM_returns)/5
totalreturn
sdfinal <- sd(totalreturn)
variance <- sdfinal^2
variance


