# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 14:10:05 2018

@author: Nikolas
"""

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from sklearn.cross_validation import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

import statsmodels.api as sm



# Importing the Data

df = pd.read_csv("winequality-red.csv", header=0, sep=";")

print 'df.shape', df.shape
df.head()


df.columns = ['fixed acidity (0)', 'volatile acidity (1)', 'citric acid (2)', 
              'residual sugar (3)', 'chlorides (4)', 'free sulfur dioxide (5)', 
              'total sulfur dioxide (6)', 'density (7)', 'pH (8)', 'sulphates (9)', 
              'alcohol (10)', 'target']
df.head()

def predict(x_i, beta):
    return np.dot(x_i, beta)

def regr_error(x_i, y_i, beta):
    return y_i - predict(x_i, beta)

def calc_pred_vals(list_of_rows, list_of_betas, intercept=0):
    pred_vals = []
    for i in range(len(list_of_rows)):
        pred_val = intercept + predict(list_of_rows[i], list_of_betas)
        pred_vals.append(pred_val)
    return pred_vals

# rmse calculation
def calc_rmse(preds, target):
    errors, squared_error = [], []
    for i in range(len(target)):
        errors.append(target[i] - preds[i])
    for error in errors:
        squared_error.append(error**2)
    mse = sum(squared_error)/len(squared_error)
    rmse = np.sqrt(mse)
    return rmse

# convert continous values to discrete
def round_prediction(predicted_values):
    preds = []
    for i in range(len(predicted_values)):
        pred_val = predicted_values[i]
        if pred_val < 0.5: preds.append(0)
        elif pred_val >= 0.5 and pred_val < 1.5: preds.append(1)
        elif pred_val >= 1.5 and pred_val < 2.5: preds.append(2)
        elif pred_val >= 2.5 and pred_val < 3.5: preds.append(3)
        elif pred_val >= 3.5 and pred_val < 4.5: preds.append(4)
        elif pred_val >= 4.5 and pred_val < 5.5: preds.append(5)
        elif pred_val >= 5.5 and pred_val < 6.5: preds.append(6)
        elif pred_val >= 6.5 and pred_val < 7.5: preds.append(7)
        elif pred_val >= 7.5 and pred_val < 8.5: preds.append(8)
        elif pred_val >= 8.5 and pred_val < 9.5: preds.append(9)
        elif pred_val >= 9.5: preds.append(10)
        else: print 'check predicted value'
    return preds

#return a list of list of confusion matrix entries
def confusion_entries(num_rows, preds, targets):
    A = [[0 for j in range(num_rows)] for i in range(num_rows)]
    #outer loop iterates through each target and prediction 
    for i in range(len(targets)): 
        pred, target = preds[i], targets[i]
        #inner loop iterates through each cell in the confusion matrix
        for j in range(num_rows): 
            for k in range(num_rows):
                #there will be one match in each inner iteration
                if target==j and pred==k: 
                    #increment value in cell when match is found
                    A[j][k] += 1  
    return A

# create dataframe for model coefficients
def make_df(betas, attributes, ix, intercept=0):
    new_beta_list, new_columns = [], []
    new_columns.append('intercept')
    new_beta_list.append(round(intercept, 5))
    for i in range(len(betas)):
        new_columns.append(attributes[i])
        beta = betas[i]
        new_beta_list.append(round(beta, 5))
    new_df = pd.DataFrame(new_beta_list, index=new_columns, columns=ix).T
    return new_df

# return requested columns of dataset - used in stepwise algorithm
def x_features_cols(dataset, features):
    x_out = []
    for row in dataset:
        x_out.append([row[i] for i in features])
    return(x_out)

# scale a row of features' values given as input for a predicted value
# receive 3 lists (row of inputs, original means & stds for each feature)
def scale_vals(row, orig_mean, orig_std):
    scaled_row = []
    for i in range(len(row)):
        scaled_value = (row[i] - orig_mean[i]) / orig_std[i]
        scaled_row.append(scaled_value)
    return scaled_row

#used in minimization function
def in_random_order(data):    
    indexes = [i for i, _ in enumerate(data)]
    random.shuffle(indexes)
    for i in indexes: 
        yield data[i]
        
def model_details(name, preds, betas, intercept, feat_seq, targets, calc_acc=True):
    
    model_name = []     
    model_name.append(name) # name must be in list format; df col gets transformed to index
    modl_name.append(name) # global variable
    
    # store rms error
    rms_error = calc_rmse(preds, targets)
    rmse.append(rms_error)
    
    # store dataframe of model's beta coefficients 
    df_coeff = make_df(betas, feat_seq, model_name, intercept)
    df_beta_coefs.append(df_coeff) 
    
    # calculate accuracy & store confusion matrix, unless labels are scaled
    if calc_acc:
        actual_ix = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']
        pred_ix = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
        rounded_preds = round_prediction(preds)
        con_mat = confusion_entries(len(pred_ix), rounded_preds, targets)
        cmat = pd.DataFrame(con_mat, index=actual_ix, columns=pred_ix)
        #delete columns with no results, eg, taste quality: 0, 1, 2, 9 & 10 are zero
        del cmat['P0']; del cmat['P1']; del cmat['P2']; del cmat['P9']; del cmat['P10']
        cmat = cmat.ix[3:] # delete rows 0, 1 & 2
        cmat = cmat.ix[:6] # delete last 2 rows
        #add column for row total
        cmat['Total']=cmat['P3']+cmat['P4']+cmat['P5']+cmat['P6']+cmat['P7']+cmat['P8']
        #denominator should equal total number of rows in dataset)
        sum_total = cmat['Total'].sum()
        # add along diagonal for true positives (numerator) 
        tp = cmat.iloc[0]['P3']+cmat.iloc[1]['P4']+cmat.iloc[2]['P5']+cmat.iloc[3]['P6'] +\
        cmat.iloc[4]['P7'] + cmat.iloc[5]['P8']
        acc = tp / sum_total  # accuracy is true positives divided by the sum total
        cmats.append(cmat)
        cmat_sum.append(sum_total)
        accuracy.append(acc)        
    else:
        cmats.append('n/a, labels are scaled')
        cmat_sum.append('n/a, labels are scaled')
        accuracy.append('n/a, labels are scaled')
        

dfvals = df.values
data_scaled = StandardScaler().fit_transform(dfvals)
boxplot(data_scaled)
plt.xlabel("Attribute Index")
plt.ylabel(("Quartile Ranges - Normalized "))
show() # run this cell prior to importing seaborn

import seaborn as sns
corr_matrx = df.corr()
sns.heatmap(corr_matrx)
plt.show()



short_col = ['fixAcid', 'volAcid', 'citAcid', 'resSugr', 'chlor', 'frSO2', 'totSO2', 'dens', 
             'pH', 'sulpha', 'alcohol', 'quality']
cormat = np.array(corr_matrx).tolist()
for i in range(len(cormat)):
    for j in range(len(cormat)):
        cormat[i][j] = round(cormat[i][j], 2)
corr_matrx = pd.DataFrame(cormat, columns=short_col, index=short_col)
corr_matrx



n_rows = len(df.index)
n_data_cols = len(df.columns) - 1
df_norm = pd.DataFrame(data_scaled, columns = short_col)
for i in range(n_rows):
    data_row = df_norm.iloc[i,1:n_data_cols]   # plot rows of data 
    norm_target = df_norm.iloc[i,n_data_cols]
    labelColor = 1.0/(1.0 + np.exp(-norm_target))
    data_row.plot(color=plt.cm.RdYlBu(labelColor), alpha=0.5)
plt.xlabel("Attribute Index")
plt.ylabel(("Attribute Values"))
plt.show()

fa = df['fixed acidity (0)']
va = df['volatile acidity (1)']
ca = df['citric acid (2)']
rs = df['residual sugar (3)']
ch = df['chlorides (4)']
fs = df['free sulfur dioxide (5)']
ts = df['total sulfur dioxide (6)'] 
de = df['density (7)']
ph = df['pH (8)'] 
su = df['sulphates (9)'] 
al = df['alcohol (10)']
qu = df['target']

fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
n, bins, patches = ax1.hist(fa)
ax1.set_xlabel('Fixed Acidity')
ax1.set_ylabel('Frequency')

fig2 = plt.figure()
ax2 = fig2.add_subplot(2, 1, 1)
n, bins, patches = ax2.hist(va)
ax2.set_xlabel('Volatile Acidity')
ax2.set_ylabel('Frequency')

fig3 = plt.figure()
ax3 = fig3.add_subplot(2, 1, 1)
n, bins, patches = ax3.hist(ca)
ax3.set_xlabel('Citric Acid')
ax3.set_ylabel('Frequency')

fig4 = plt.figure()
ax4 = fig4.add_subplot(2, 1, 1)
n, bins, patches = ax4.hist(rs)
ax4.set_xlabel('Residual Sugar')
ax4.set_ylabel('Frequency')

fig5 = plt.figure()
ax5 = fig5.add_subplot(2, 1, 1)
n, bins, patches = ax5.hist(ch)
ax5.set_xlabel('Chlorides')
ax5.set_ylabel('Frequency')

fig6 = plt.figure()
ax6 = fig6.add_subplot(2, 1, 1)
n, bins, patches = ax6.hist(fs)
ax6.set_xlabel('Free Sulfur Dioxide')
ax6.set_ylabel('Frequency')

fig7 = plt.figure()
ax7 = fig7.add_subplot(2, 1, 1)
n, bins, patches = ax7.hist(ts)
ax7.set_xlabel('Total Sulfur Dioxide')
ax7.set_ylabel('Frequency')

fig8 = plt.figure()
ax8 = fig8.add_subplot(2, 1, 1)
n, bins, patches = ax8.hist(de)
ax8.set_xlabel('Density')
ax8.set_ylabel('Frequency')

fig9 = plt.figure()
ax9 = fig9.add_subplot(2, 1, 1)
n, bins, patches = ax9.hist(ph)
ax9.set_xlabel('pH')
ax9.set_ylabel('Frequency')

fig10 = plt.figure()
ax10 = fig10.add_subplot(2, 1, 1)
n, bins, patches = ax10.hist(su)
ax10.set_xlabel('Sulphates')
ax10.set_ylabel('Frequency')

fig11 = plt.figure()
ax11 = fig11.add_subplot(2, 1, 1)
n, bins, patches = ax11.hist(al)
ax11.set_xlabel('Alcohol')
ax11.set_ylabel('Frequency')

fig12 = plt.figure()
ax12 = fig12.add_subplot(2, 1, 1)
n, bins, patches = ax12.hist(qu)
ax12.set_xlabel('Taste')
ax12.set_ylabel('Frequency')


stats = df.describe()
stats


limits_upper, limits_lower = [], []
max_vals, min_vals = [], [] 
outliers_upper, outliers_lower = [], []
num_upper, num_lower = [], []

#loop through features - get data on outliers
for i in range(len(stats.columns)-1):
    
    # set interquartile range
    inter_quartile_range = stats.iloc[6,i] - stats.iloc[4,i]
    
    # upper limits
    limit_upper = stats.iloc[6,i] + 1.5 * inter_quartile_range
    outliers_upper.append(df.loc[df[df.columns[i]] > limit_upper])
    limits_upper.append(limit_upper)
    max_vals.append(stats.iloc[7,i])
    
    # lower limits
    limit_lower = stats.iloc[4,i] - 1.5 * inter_quartile_range
    outliers_lower.append(df.loc[df[df.columns[i]] < limit_lower])
    limits_lower.append(limit_lower)
    min_vals.append(stats.iloc[3,i])
    
    
feature_cols = df.columns[0:11]
for i in range(len(stats.columns)-1): 
    num_upper.append(len(outliers_upper[i]))
df_out_up = pd.DataFrame(zip(feature_cols, limits_upper, max_vals, num_upper), 
                         columns = ['Feature', 'Upper Limit', 'Max Value', '# Outliers'])
df_out_up


for i in range(len(stats.columns)-1): 
    num_lower.append (len(outliers_lower[i]))
df_out_low = pd.DataFrame(zip(feature_cols, limits_lower, min_vals, num_lower), 
                          columns = ['Feature', 'Lower Limit', 'Min Value', '# Outliers'])
df_out_low

print 'df shape', df.shape
for i in range(len(stats.columns)-1):
    df = df.loc[df[feature_cols[i]] < limits_upper[i]]
    df = df.loc[df[feature_cols[i]] > limits_lower[i]]
print 'df shape after removing rows with outliers', df.shape


x_list = np.array(df[list(df.columns)]).tolist()
cols = df.columns
df = pd.DataFrame(x_list, columns=cols)
df.tail()




x = df[list(df.columns)[:-1]]
y = df['target']
x = sm.add_constant(x)  #include intercept, since x is not scaled
modl = sm.OLS(y, x).fit()  #y goes before x in statsmodels
modl.summary()



# 11 features, excluding labels, in list of lists format
x_list = np.array(df[list(df.columns)[:-1]]).tolist()

# labels in list format
labels = df['target'].tolist()

# 12 features, scaled
dfvals = df.values
data_scaled = StandardScaler().fit_transform(dfvals)

# 11 features, scaled, in dataframe format
# lables, scaled, in list format
df_scaled = pd.DataFrame(data_scaled, columns=df.columns)
labels_scaled = df_scaled['target'].tolist()
del df_scaled['target']

# 11 features, scaled, in list of lists format
x_scaled = np.array(df_scaled).tolist()


df.target.value_counts()









### 4

n_rows = len(df.index)
n_cols = len(x_list[0])
beta = [0.0] * n_cols # row of beta coefficients
beta_matrix = [] # matrix of beta coefficients
beta_matrix.append(list(beta))
n_steps = 350
step_size = 0.004
lars_indices = [] # features placement sequence

# calculate correlation between features & residuals
# and increment (decrement) beta corresponding to feature with highest correlation 
for i in range(n_steps):
    residuals = [0.0] * n_rows
    for j in range(n_rows):
        labels_hat = sum([x_scaled[j][k] * beta[k] for k in range(n_cols)])
        residuals[j] = labels_scaled[j] - labels_hat
    corr = [0.0] * n_cols    
    for j in range(n_cols):
        corr[j] = sum([x_scaled[k][j] * residuals[k] for k in range(n_rows)]) / n_rows
    k = 0    
    corr_i = corr[0]
    for j in range(1, (n_cols)):
        if abs(corr_i) < abs(corr[j]):
            k = j
            corr_i = corr[j]
    beta[k] += step_size * corr_i / abs(corr_i)
    beta_matrix.append(list(beta))
    beta_indices = [index for index in range(n_cols) if beta[index] != 0.0]
    for q in beta_indices:
        if (q in lars_indices) == False:
            lars_indices.append(q)

#find the beta matrix index with lowest rmse
errors = []
#x = np.array(df_scaled).tolist()  #list of list format for calc_pre_vals
#y = labels_scaled
for i in range(len(beta_matrix)):
    preds = calc_pred_vals(x_scaled, beta_matrix[i])
    rms_error = calc_rmse(preds, labels_scaled)
    errors.append(rms_error)
min_error = min(errors)
min_index = errors.index(min_error)
print 'index of betas with lowest rmse', min_index, '  minimum rmse', min_error
    
#plot range of beta values for each attribute
for i in range(n_cols):
    coeff_curve = [beta_matrix[k][i] for k in range(n_steps)]
    xaxis = range(n_steps)
    plt.plot(xaxis, coeff_curve)
print '\ncoefficients in original feature order'
dfnames = df.columns
coeffs = beta_matrix[min_index]
print (zip(dfnames, coeffs))  
lars_sequence = [dfnames[lars_indices[i]] for i in range(len(lars_indices))]  
lars_coeffs = [coeffs[i] for i in lars_indices]
print '\nfeature priority under LARS algorithm\n', lars_sequence
plt.xlabel("Steps Taken")
plt.ylabel(("Coefficient Values"))
plt.show()

##### 5

def run_lars_5(x, y):
    preds = calc_pred_vals(x, lars_coeffs)  # lars_coeffs is global; no intercept
    model_details('Lars, from scratch, xy scaled', preds, lars_coeffs, 0, lars_sequence, 
                  y, False)
    
    
x = df_scaled[lars_sequence]
y = labels_scaled
alpha_list = [0.001, 0.01, 0.1]  
error_list = []
for alph in alpha_list:
    modl = Lasso(alpha=alph, normalize=False) 
    modl.fit(x, y)   
    preds = cross_val_predict(modl, x, y, cv=10) #make predictions
    rms_error = calc_rmse(preds, y)  #calculate errors
    error_list.append(rms_error)
print 'x and y scaled'
print("RMS Error             alpha")
for i in range(len(error_list)):
    print(error_list[i], alpha_list[i])

y = df['target']
alpha_list = [0.001, 0.01, 0.1]  
error_list = []
for alph in alpha_list:
    modl = Lasso(alpha=alph, normalize=False) 
    modl.fit(x, y)   
    preds = cross_val_predict(modl, x, y, cv=10) #make predictions
    rms_error = calc_rmse(preds, y)  #calculate errors
    error_list.append(rms_error)
print '\nx scaled  (labels not scaled)'
print("RMS Error             alpha")
for i in range(len(error_list)):
    print(error_list[i], alpha_list[i])
    
    
def run_lasso_7(x, y):
    modl = Lasso(alpha=0.01, normalize=False)
    modl.fit(x,y)
    preds = cross_val_predict(modl, x, y, cv=10) #make predictions
    model_details('Lasso alpha 0.01, sklearn, xy scaled', preds, modl.coef_, 
                  modl.intercept_, lars_sequence, y, False)
    
def run_lasso_8(x, y):
    modl = Lasso(alpha=0.01, normalize=False)
    modl.fit(x,y)
    preds = cross_val_predict(modl, x, y, cv=10) #make predictions
    model_details('Lasso alpha 0.01, sklearn, x scaled', preds, modl.coef_, 
                  modl.intercept_, lars_sequence, y)