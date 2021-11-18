from sklearn import linear_model,metrics
import pandas as pd
import numpy as np
from sklearn import model_selection

df = pd.read_csv( 'Stock Performance.csv')


annret = df.loc[:,"Annual Return"]

X=df[['Large BP', 'Large ROE', 'Large SP', 'Large Return Rate',
       'Large Market Value', 'Small Systematic Risk']]
Y=df[["Annual Return"]]

# Split the training data from the test data
X_train, X_test, Y_train, Y_test = model_selection.train_test_split( X, Y, test_size = 0.80 )


#Linear Regression Model
regr = linear_model.LinearRegression()

regr.fit(X_train, Y_train)

Y_pred = regr.predict(X_test)

print("Linear")
# Show the errors
print('MAE:' + str(metrics.mean_absolute_error(Y_test,Y_pred)))
print('MSE:' + str(metrics.mean_squared_error(Y_test,Y_pred)))
print('RMSE:'+ str(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))))
#Show the R-Squared
print('R-SQUARED:'+str(metrics.r2_score(Y_test,Y_pred)))

#Ridge Regression Model
regr = linear_model.Ridge()

regr.fit(X_train, Y_train)

Y_pred = regr.predict(X_test)
print("Ridge")
# Show the errors
print('MAE:' + str(metrics.mean_absolute_error(Y_test,Y_pred)))
print('MSE:' + str(metrics.mean_squared_error(Y_test,Y_pred)))
print('RMSE:'+ str(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))))
#Show the R-Squared
print('R-SQUARED:'+str(metrics.r2_score(Y_test,Y_pred)))


#Elastic Net Regression Model
regr = linear_model.ElasticNet()

regr.fit(X_train, Y_train)

Y_pred = regr.predict(X_test)
print("Elastic Net")
# Show the errors
print('MAE:' + str(metrics.mean_absolute_error(Y_test,Y_pred)))
print('MSE:' + str(metrics.mean_squared_error(Y_test,Y_pred)))
print('RMSE:'+ str(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))))
#Show the R-Squared
print('R-SQUARED:'+str(metrics.r2_score(Y_test,Y_pred)))


#Lasso Regression Model
regr = linear_model.Lasso()

regr.fit(X_train, Y_train)

Y_pred = regr.predict(X_test)
print("Lasso")

# Show the errors
print('MAE:' + str(metrics.mean_absolute_error(Y_test,Y_pred)))
print('MSE:' + str(metrics.mean_squared_error(Y_test,Y_pred)))
print('RMSE:'+ str(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))))
#Show the R-Squared
print('R-SQUARED:'+str(metrics.r2_score(Y_test,Y_pred)))