import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pickle
from pickle import dump
import sklearn
from sklearn.datasets import load_boston



boston = load_boston()
bos = pd.DataFrame(boston.data,columns = boston.feature_names)


#Removing outliers from PTRATIO
upper_boundary = bos['PTRATIO'].mean() + 1.5*bos['PTRATIO'].std()
lower_boundary = bos['PTRATIO'].mean() - 1.5*bos['PTRATIO'].std()
#print(lower_boundary),print(upper_boundary),print(bos['PTRATIO'].mean())
data = bos.copy()
#data['target'] = boston.target

data.loc[data['PTRATIO']<lower_boundary,'PTRATIO'] = lower_boundary


#Removing outliers from CRIM
IQR = bos.CRIM.quantile(0.75) - bos.CRIM.quantile(0.25)
lower_bridge = bos['CRIM'].quantile(0.25) - (IQR*3)
upper_bridge = bos['CRIM'].quantile(0.75) + (IQR*3)
data.loc[data['CRIM']>=upper_bridge,'CRIM'] = upper_bridge

#Removing outliers from ZN
IQR = bos.ZN.quantile(0.75) - bos.ZN.quantile(0.25)
lower_bridge = bos['ZN'].quantile(0.25) - (IQR*1.5)
upper_bridge = bos['ZN'].quantile(0.75) + (IQR*1.5)
data.loc[data['ZN']>=upper_bridge,'ZN'] = upper_bridge 

#Removing outliers from DIS
IQR = bos.DIS.quantile(0.75) - bos.DIS.quantile(0.25)
lower_bridge = bos['DIS'].quantile(0.25) - (IQR*1.5)
upper_bridge = bos['DIS'].quantile(0.75) + (IQR*1.5)
data.loc[data['DIS']>=upper_bridge,'DIS'] = upper_bridge 

#Removing outliers from B
IQR = bos.B.quantile(0.75) - bos.B.quantile(0.25)
lower_bridge = bos['B'].quantile(0.25) - (IQR*1.5)
upper_bridge = bos['B'].quantile(0.75) + (IQR*1.5)
data.loc[data['B']<=lower_bridge,'B'] = lower_bridge 

#Dropping columns TAX & RAD which have high multicollinearity and Age and Indus which are not very significant(p-values)
data.drop(['AGE','INDUS','TAX','RAD'],axis = 1,inplace = True)

import scipy.stats as stat
import pylab

def plot_data(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.subplot(1,2,2)
    stat.probplot(df[feature],dist='norm',plot=pylab)
    plt.show()

data['CRIM_log'] = np.log(data['CRIM'])
#plot_data(data,'CRIM_log')

data.drop(['CRIM'],axis = 1,inplace = True)

print(data.columns)

X = data.loc[:,['ZN', 'CHAS', 'NOX', 'RM', 'DIS', 'PTRATIO', 'B', 'LSTAT','CRIM_log']]
y = boston.target

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler =StandardScaler()

X_scaled = scaler.fit_transform(X)

x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.25,random_state=355)


from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import  LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


regression = LinearRegression()
regression.fit(x_train,y_train)

regression.score(x_train,y_train)

def adj_r2(x,y):
    r2 = regression.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-((1-r2)*(n-1)/(n-p-1))
    return adjusted_r2

adj_r2(x_train,y_train)

regression.score(x_test,y_test)

adj_r2(x_test,y_test)

# saving the model to the local file system
filename_2 = 'finalized_model.pickle'
pickle.dump(regression, open(filename_2, 'wb'))

# saving the scaler to the local file system
filename_1 = 'scaler.pickle'
pickle.dump(scaler, open(filename_1, 'wb'))


# prediction using the saved model
loaded_model = pickle.load(open(filename_2, 'rb'))
a=loaded_model.predict(scaler.transform([[0,0.0,0.573,6.030,2.5050,21.0,396.9,7.88,0.04741]]))
print('the predicted value is :',a)
a