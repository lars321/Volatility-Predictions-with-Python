
# coding: utf-8

# In[1]:


#Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import sklearn
import pandas_datareader.data as web
import datetime as dt
from statsmodels.graphics.tsaplots import plot_acf

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


###Functions for estimators###
def forecast(raw_data,model,forecast_horion):
    '''
    Function for aggregated volatility dependend variable:

    Parameters
    ----------
    raw_data : Dataframe with one row
        raw Data
    model : Dataframe
        Should be Dataframe to run the regression in a later step
    forecast_horion : int
        aggregated volatility, e.g. 2 for average (agregated) volatility in the next two days
    
    Returns
    -------
    Dataframe
        model with additional column containg the forecast 
    '''
    def moving_average(a): #rolling function
        f1=sum(a[:])/(len(a)) #average of all the previous realizations excl. today (a[-1])
        return f1
    
    label="RV^(%d)_t"%(forecast_horion)
    raw_data_reveresed=raw_data[::-1] #reverse order to apply rolling function
    raw_data_reveresed[label] = raw_data_reveresed.rolling(forecast_horion).apply(moving_average)
    model[label]=raw_data_reveresed[label]#apply rolling function to 
    return model


def lag_estimator(raw_data,model,lag): #create new column with lag estimator
    '''
    Function for lag volatility estimators:

    Parameters
    ----------
    raw_data : Dataframe with one row
        raw Data
    model : Dataframe
        Should be Dataframe to run the regression in a later step
    lag : int
        laged volatility estimator, e.g. 3 for a three day lag
    
    Returns
    -------
    Dataframe
        model with additional column containg the estimator 
    '''
    def lag_value(a): #rolling function
        f1=a[0] #average of all the previous realizations excl. today (a[-1])
        return f1

    label="RV_t-%d"%lag
    #apply rolling function
    model[label] = raw_data.iloc[:,0].rolling(lag+1).apply(lag_value)   
    return model

def har_estimator(raw_data,model,horrizon): 
    '''
    Function for aggregated volatility estimators:

    Parameters
    ----------
    raw_data : Dataframe with one row
        raw Data
    model : Dataframe
        Should be Dataframe to run the regression in a later step
    estimators_horizon : int
        aggregated volatility estimator, e.g. 22 for monthly component
    
    Returns
    -------
    Dataframe
        model with additional column containg the estimator 
    '''
    def moving_average(a): #rolling function
        f1=sum(a[:-1])/horrizon #average of prev. realizations excl. today
        return f1

    label="RV^(%d)"%horrizon #create Lable
    #apply rolling function 
    model[label] = raw_data.iloc[:,0].rolling(horrizon+1).apply(moving_average) 
    return model


# In[3]:


def AR_model(raw_data,forecast_horizon,estimators_horizon):
    '''
    Function for Autoregresive model:

    Parameters
    ----------
    raw_data : Dataframe with one row
        raw Data
    forecast_horizon : int
        Forecast horizon of independent varaibles, e.g. 1
    estimators_horizon : list with int
        lag estimators, e.g. [1,2,3] for one day, two day, three day lags 
    
    Returns
    -------
    Dataframe
        Dataframe with first row dependend variables 
        and following rows independend variables
    '''
    model=pd.DataFrame(index=raw_data.index)
    forecast(raw_data,model,forecast_horizon) #1st row with dependend variables

    for i in estimators_horizon:
        lag_estimator(raw_data,model,i)
        
    return model

def har_model(raw_data,forecast_horizon,estimators_horizon):
    '''
    Function for HAR model:

    Parameters
    ----------
    raw_data : Dataframe with one row
        raw Data
    forecast_horizon : int
        Forecast horizon of independent varaibles, e.g. 1
    estimators_horizon : list with int
        aggregated volatility estimators, 
        e.g. [1,5,22] for daily, weekly and monthly component
    
    Returns
    -------
    Dataframe
        Dataframe with 1st row dependend var. and independend var.
    '''
    model=pd.DataFrame(index=raw_data.index) #set DataFrame with index
    forecast(raw_data,model,forecast_horizon) #dependend variables

    for i in estimators_horizon: #Independent variables variables
        har_estimator(raw_data,model,i)

    return model


# In[4]:


def remove_NaN(models):
    '''
    Drop any rows with NaN
    
    The models still contain NaN in the rows,  
    as for example har_estimator function is not defined for first 22 rows.
    Therefore we need to exclude all rows with NaN.
    To make the models comparable, we have delete the rows in all the models
    
    Parameters
    ----------
    models : list with Dataframes
        list of all the models, e.g. [model1,model2,model3]
    
    Returns
    -------
    list with Dataframes
        Models without any NaN
    '''
    
    lst=[] #List of rownumbers with NaNs

    for item1 in models: #Find the rows with NaNs
        i=0
        for item2 in item1.isnull().any(axis=1):
            if item2 == True and i not in lst:
                lst.append(i)
            i+=1

    for item in models: #Drop the rows with NaNs
        item.drop(item.index[[lst]],inplace=True)
    return models


# In[5]:


def regress_model(model):
    '''
    Regress the models
    '''
    Y = model.iloc[:,0]
    X = model.iloc[:,1:]
    X = sm.add_constant(X)
    reg = sm.OLS(Y,X)
    results = reg.fit()
    return results


# In[6]:


def test_model(model, training_window,reg):
    '''
    Function for training the model and then test it

    I test the models applying a rolling-forecast manner, 
    updating the transform and model for each time step, see 
    https://machinelearningmastery.com/time-series-forecast-study-python-annual-water-usage-baltimore/
    
    
    Parameters
    ----------
    model : Dataframe
        model definded as above with first row dependend variables 
        and following rows independed variabels
    training_window : int
        number of periods in the training window, should be shorter than the model
    reg : sklearn.linear_model
        Can be a Lasso regression or ols regression
        
    Returns
    -------
    Int
        Root Mean Squared Error of the predictions, the smaller the better
    list
        Predictions made
    list
        Ture values
    '''
    training_testing_window=[training_window,model.shape[0]-training_window] 
    #define a vector with training window and testing window
    
    #Load the model and use it in a rolling-forecast manner, updating the transform and model for each time step.
    predictions=[]
    for i in range(0,training_testing_window[1]):
        y_train = model.iloc[i:training_testing_window[0]+i,0]
        X_train = model.iloc[i:training_testing_window[0]+i,1:]
        X_test = np.array([model.iloc[training_testing_window[0]+i,1:]])
        
        model_fit=reg.fit(X_train, y_train)
        model_prediction=model_fit.predict(X_test)[0]
        predictions.append(model_prediction)
    
    #put predicitons into a dataframe
    predictions = pd.DataFrame(predictions, index=model.iloc[training_testing_window[0]:,0].index) 
    Observed_values = model.iloc[training_testing_window[0]:,0]
    
    #Calculate mean squared error of predictons
    mse = mean_squared_error(Observed_values, predictions)
    #Take the Root means squared error to evaluate the forecast performance
    rmse = sqrt(mse)
    
    #Ploting Predcitions vs. Realizations
    

    return rmse,predictions,Observed_values


# In[7]:


#load data
start = dt.datetime(2014,10,1)
end = dt.datetime(2018,10,1)

#I use the Amazon stocks
df=web.DataReader('AMZN','iex', start, end).reset_index()
df['date']=pd.DatetimeIndex(df['date'])
df = df.set_index(df['date'],drop=True)

'''Calcualte Garman Klass Volatility, see: 
https://breakingdownfinance.com/finance-topics/risk-management/garman-klass-volatility/'''
df["volatility"]=np.power(                   1/2*np.power(np.log(df["high"]/df["low"]),2)                   -(2*np.log(2)-1)*np.power(np.log(df["close"]/df["open"]),2)                   ,0.5)
raw_data=pd.DataFrame(df["volatility"])

#Descriptive Statistics
print(raw_data.describe())

#Time Series Plot
plt.plot(raw_data)
plt.title("Volatility")
plt.gcf().autofmt_xdate()
plt.show()

#Histogram
plt.hist(raw_data.iloc[:,0])
plt.title("Histogram")
plt.show()

#Autocorrelation Plot
plot_acf(raw_data.iloc[:,0],lags=50)
plt.show()


# In[8]:


#Calculate three different models
AR_model_3=AR_model(raw_data,1,[1,2,3])
HAR_model=har_model(raw_data,1,[1,5,22])
AR_model_22=AR_model(raw_data,1,range(1,23))


models = [AR_model_3,HAR_model,AR_model_22]
labels = ["AR(3) Model","HAR(1,5,22) Model","AR(22) Model with OLS","AR(22) Model with Lasso"]

models=remove_NaN(models)
    
#print OLS regression tables
i=0
for item in models:
    print(labels[i],"\n")
    print(regress_model(item).summary(),"\n\n")
    i+=1
    
#print Out of sample performance and tables
RMSE_for_models=pd.DataFrame([],index=labels,columns=["RMSE"]) #list of all the RMSE

reg_OLS=sklearn.linear_model.LinearRegression()
training_window=int((len(AR_model_3)/2)) #set training window as half the sample size

plt.close()
fig = plt.figure(figsize=(8,8))

for item,i in zip(models,range(0,3)) :
    #calculate modles
    rmse, predictions,Observed_values=test_model(item,training_window,reg_OLS) 
    RMSE_for_models["RMSE"][i]=rmse #set RMSE

    #Plot predictions vs. observed values
    ax = fig.add_subplot(4,1,i+1)
    ax.plot(Observed_values)
    ax.plot(predictions,color="red")
    ax.set_title(labels[i])
    ax.tick_params()

#estimate AR(22) with Lasso
reg_Lasso=sklearn.linear_model.LassoCV()
rmse, predictions,Observed_values=test_model(AR_model_22,training_window,reg_Lasso)

RMSE_for_models["RMSE"][3]=rmse #set RMSE

ax = fig.add_subplot(4,1,4)
ax.plot(Observed_values)
ax.plot(predictions,color="red")
ax.set_title("AR(22) Model with Lasso")
ax.tick_params()
 
plt.tight_layout()
plt.show()

print("The models have the following RMSE (lower is better):\n\n",RMSE_for_models)
best_model=pd.DataFrame.idxmin(RMSE_for_models.apply(pd.to_numeric, errors = 'coerce', axis = 0))[0]
print("\nThe %s preforms best in the out of sample rolling-forecasts.\n"%best_model)

