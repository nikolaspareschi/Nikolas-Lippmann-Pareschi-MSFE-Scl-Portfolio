
# Calculate stock price at time t+1
#  Input
#  st = stock price at time t
#  r = expected annual stock return
#  sigma = annualized volatility in prices
#  t = time in years
#  n = number of steps
#  e = epsilon


# Project guideline 1. Consider following values for the purpose of this project:

# 
# It was given that St is the stock at price t (St = 10)
# S t+1 is the stock at price t+1
# r is the expected annual stock return, (Say 0.15)
# sd is thr annualized volatility of underlying stock (0.2)
# T is time in years (1 year)
# n is the number of steps inveolved in the calculation (12 steps)
# deltat is the size of the unit step size = T/n
# epsilon is the distribution term with a zero mean, (a random value from a normal sample) (epsilon = 0.15)




# Now we consider the inputs given


st = 10 
r = 0.15 
sigma = 0.2 
t = 1 
n = 100 
e = 0.15


# Project guideline 2. Starting with the initial stock price St as specified, and considering 100 Steps,
# calculate the expected value of the stock price at the end of every successive ??t interval of time


# First we need to define the function given in case 3 for the price



st1 = function(st, r, sigma, t, n, e) {
  dt = t/n 
  return(st * exp(((r - (0.5 * (sigma ^ 2))) * dt) + (sigma * e * sqrt(dt))))
}


# Calculating the expected value of the stock price at the end of every successive deltat interval of time

evarray = function(st, r, sigma, t, n, e) {
  ev = c()
  ev[1] = st
  for (i in 1:n) {
    
    ev[i + 1] = st1(ev[i], r, sigma, t, n, e)
  }
  return(ev)
}


ev = evarray(st, r, sigma, t, n, e)
ev


#  Project guideline 3. Plot the entire movement of prices over the T period under observation


plot(ev,
     main = 'Movement of prices with st = 10, r = 0.15, sigma = 0.2, t = 1, n =100, e = 0.15 ',
     xlab = 'n', 
     ylab = 'Prices', 
     xlim = c(0, 100), 
     ylim = c(0, 20), 
     type='l',cex.lab=1, cex.axis=1, cex.main=0.8, cex.sub=1)


#  Project guideline  4. Instead of considering a fixed ?? as in the previous steps, randomly assign values to ?? from a standard normal distribution.


evarray2 = function(st, r, sigma, t, n, randome) {
  ev2 = c()
  ev2[1] = st
  for (i in 1:n) {
      epsilon = rnorm(1)
      ev2[i + 1] = st1(ev2[i], r, sigma, t, n, epsilon)
  }
  return(ev2)
}

ev2 = evarray2(st, r, sigma, t, n, randome)
ev2






plot(ev2, 
     main = 'Random epsilon in normal distribution in each time step',
     xlab = 'Step', 
     ylab = 'Stock Price', 
     xlim = c(0, 100), 
     ylim = c(0, 20), 
     type='l')

#  Project guideline  5.  Perform 5 trials of 100 steps each to plot 
# the probable movement of stock prices over a 1 year period. Plot each trajectory of prices as a separate line

colors = rainbow(5)
plot(
  c(0, 100),
  c(0, 20), 
  main = '5 trials of 100 steps each with standard normal distributed Epsilon',
  xlab = 'Step',
  ylab = 'Stock Price',
  type='n',cex.lab=1, cex.axis=1, cex.main=0.8, cex.sub=1)

for (i in 1:5) {
  ev2 = evarray2(st, r, sigma, t, n, epsilon)
  lines(ev2, col = colors[i])
  text(90, ev2[101], paste('i', i), cex = 1.0, col = colors[i])
}


