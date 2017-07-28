import matplotlib
import sklearn
import  numpy
import pandas as pd
import numpy as np
import quandl
import math, datetime
from sklearn import preprocessing,cross_validation,svm,linear_model
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

print('OM GAN GANAPATHAYE NAMAHA')

quandl.ApiConfig.api_key = 'xbPnXMAxKyEVqwzW9TWv'
df  = quandl.get('WIKI/GOOGL')
print(df.tail())
df['HL_PCT'] =(df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_change'] =(df['Adj. Close']-df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
forecast_col = df['Adj. Close']
forecast_out = int(math.ceil(0.001*len(forecast_col)))
df['label'] = forecast_col.shift(-forecast_out)
df.dropna(inplace=True)

X=np.array(df.drop(['label'],1))#,'adj_close'],1))

X=preprocessing.scale(X)
#print("X after preprocessing.scale ",X)
X_lately = X[-forecast_out:]
#print("X_lately",X_lately)
X=X[:-forecast_out]

#print(df)
#print("X",X)
Y=np.array(df['label'])
#Y=preprocessing.scale(Y)
Y=Y[:-forecast_out]
#print("Y ",Y)

x_train,x_test,y_train,y_test=cross_validation.train_test_split(X,Y, test_size=0.2)
clf=linear_model.LinearRegression(n_jobs=-1)
clf=clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test) 
forecast_set=clf.predict(X_lately)
#forecast_set_whole=clf.predict(X)
print(accuracy,forecast_set[0])#,forecast_set_whole)
cl=KNeighborsRegressor()
cl.fit(x_train,y_train)
pr=cl.predict(X_lately)
accu=cl.score(x_test,y_test)
print(accu,pr[0])
#forecast_set=np.array(forecast_set)
#df['Forecast'] = np.nan

#last_date=df.iloc[-1].name
#last_unix=last_date#.timestamp()
#one_day=86400
#next_unix=last_unix+one_day

#for i in forecast_set:
 #next_date = next_unix#datetime.datetime.fromtimestamp(next_unix)
 #next_unix+=one_day
 #df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]

#print(df.label[:-forecast_out])
#print(forecast_set)
#df['adj_close'].plot()
#df['Forecast'].plot()
#plt.legend(loc=4)
#plt.xlabel('Date')
#plt.ylabel('Price')
#plt.show()
