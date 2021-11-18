                                    #Q.1#

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import yfinance as yf

#Set start & end date
start_date='2016-01-04'
end_date='2018-07-31'

#Download csv from yfinance
SP500=yf.download('^GSPC', start=start_date, end=end_date)
SP500.to_csv('SP500' +'_'+'April16-July18.csv')

#Load data into pandas dataframe
SP500=pd.read_csv('SP500' +'_'+'April16-July18.csv')
#print(SP500)

# Read a .csv file into a dataframe
MCD = pd.read_csv('MCD.csv')
#print(MCD)

#Compute the daily return
MCD=np.array(MCD.Close.pct_change()[1:])
SP500=np.array(SP500.Close.pct_change()[1:])
#print(MCD)
#print(SP500)

# Calculate Beta using the covariance
covariance = np.cov( MCD, SP500 )
beta = covariance[ 0, 1 ] / covariance[ 1, 1 ]
print('The Beta value using CoVariance is '+str(beta))


# Calculate Beta using linear regression
model = LinearRegression()
model.fit( SP500.reshape( -1, 1 ), MCD )
print('The Beta value using Linear Regression is -- '+str(model.coef_[0]))












