""" importing libraries """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


""" importing datasets """
dataset1 = pd.read_csv("Hackthon_case_training_data.csv")
dataset2 = pd.read_csv("Hackathon_case_training_hist_data.csv")
dataset3 = pd.read_csv("Hackthon_case_training_output.csv")


"""     TASK 1 :    FINDING VARIABLES INDICATING CHURN      """


""" checking columns with null values in datasets """
print(dataset1.isnull().sum())
print(dataset2.isnull().sum())

""" dropping column which had all enties as null """
dataset1.drop('campaign_disc_ele', axis=1, inplace=True)

""" checking coorelation between independent variables """
test = dataset1
test = test.dropna(axis=0, subset=['forecast_cons'])
print(test['forecast_cons'].corr(test['imp_cons']))

""" filling missing values on the basis of coorelation """
dataset1['forecast_cons']=dataset1['imp_cons']

""" dropping rows which had null value  """
dataset1 = dataset1.dropna(axis=0, subset=['date_end'])
dataset1 = dataset1.dropna(axis=0, subset=['pow_max'])
dataset1 = dataset1.dropna(axis=0, subset=['pow_max'])

""" Converting dates into days to include into dataset """
dataset1['date_activ'] = pd.to_datetime(dataset1['date_activ'])
dataset1['date_end'] = pd.to_datetime(dataset1['date_end'])
dataset1['contract_duration'] = (dataset1['date_end'] - dataset1['date_activ']).dt.days

""" dropping columns which do not appear to be influencing churn output much """
dataset1.drop('activity_new',axis=1, inplace=True)
dataset1.drop('date_first_activ',axis=1, inplace=True)
dataset1.drop('date_modif_prod',axis=1, inplace=True)
dataset1.drop('date_renewal',axis=1, inplace=True)
dataset1.drop('date_activ',axis=1, inplace=True)
dataset1.drop('date_end',axis=1, inplace=True)
dataset1.drop('forecast_base_bill_ele',axis=1, inplace=True)
dataset1.drop('forecast_base_bill_year',axis=1, inplace=True)
dataset1.drop('forecast_bill_12m',axis=1, inplace=True)

""" checking unique """
print(dataset1['channel_sales'].value_counts(),"\n")
print(dataset1['origin_up'].value_counts(),"\n")
print(dataset1['forecast_discount_energy'].value_counts(),"\n")
print(dataset1['net_margin'].value_counts(),"\n")  
print(dataset1['margin_net_pow_ele'].value_counts(),"\n") 
print(dataset1['margin_gross_pow_ele'].value_counts(),"\n")

""" filling most frequent in place of missing values """
dataset1['channel_sales'] = dataset1['channel_sales'].fillna(dataset1['channel_sales'].value_counts().index[0])
dataset1['origin_up'] = dataset1['origin_up'].fillna(dataset1['origin_up'].value_counts().index[0])
dataset1['forecast_discount_energy'] = dataset1['forecast_discount_energy'].fillna(dataset1['forecast_discount_energy'].value_counts().index[0]) #filled 0
dataset1['net_margin'] = dataset1['net_margin'].fillna(dataset1['net_margin'].value_counts().index[0])
dataset1['margin_net_pow_ele'] = dataset1['margin_net_pow_ele'].fillna(dataset1['margin_net_pow_ele'].value_counts().index[0])
dataset1['margin_gross_pow_ele'] = dataset1['margin_gross_pow_ele'].fillna(dataset1['margin_gross_pow_ele'].value_counts().index[0])

""" filling zero because the customer appears inactive last month """
dataset1['forecast_price_energy_p1'].fillna(0,inplace = True)
dataset1['forecast_price_energy_p2'].fillna(0,inplace = True)
dataset1['forecast_price_pow_p1'].fillna(0,inplace = True)


"""    TASK 2  :  finding relation between subscribed power and consumption   """


x = dataset1.iloc[:,14].values
y = dataset1.iloc[:,21].values
fig,axis=plt.subplots()
axis.scatter(x,y)
plt.title('subscribed power vs consumption')
plt.xlabel('consumption')
plt.ylabel('subscribed power')
plt.show()

"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x, color = 'blue'))
plt.title('subscribed power vs consumption')
plt.xlabel('consumption')
plt.ylabel('subscribed power')
plt.show()
"""

""" filling top values in successive rows """
dataset2 = dataset2.fillna(method='ffill')

""" adding dataset2 in dataset1 for including historical data in our dataset """
dataset1 = pd.merge(dataset1,dataset2.groupby('id').agg({'price_p1_var': np.mean,'price_p2_var':np.mean,'price_p3_var':np.mean,'price_p1_fix':np.mean,'price_p2_fix':np.mean,'price_p3_fix':np.mean}),on='id')  

""" converting object type to dataframe """
temp = dataset1['id']
temp = pd.DataFrame(temp)
dataset1 = dataset1.iloc[:,1:].values

#Z = pd.DataFrame(dataset1)

""" Encoding categorical variables """
labelencoder_x_1 = LabelEncoder()
dataset1[:,0] = labelencoder_x_1.fit_transform(dataset1[:,0])
labelencoder_x_2 = LabelEncoder()
dataset1[:,12] = labelencoder_x_2.fit_transform(dataset1[:,12])
labelencoder_x_3 = LabelEncoder()
dataset1[:,19] = labelencoder_x_3.fit_transform(dataset1[:,19])

#Z = pd.DataFrame(X)
Z = pd.DataFrame(dataset1)

#Encoding categorical variables so that none has priority over the other

onehotencoder1 = OneHotEncoder(categorical_features = [0])
dataset1 = onehotencoder1.fit_transform(dataset1).toarray()
dataset1 = dataset1[:, 1:]
Z = pd.DataFrame(dataset1)
onehotencoder3 = OneHotEncoder(categorical_features = [24])
dataset1 = onehotencoder3.fit_transform(dataset1).toarray()
dataset1 = dataset1[:, 1:]
dataset1 = pd.DataFrame(dataset1)

dataset1 = pd.concat([temp.reset_index(drop=True),dataset1.reset_index(drop=True)],axis=1)

#dataset1["id"] = temp.iloc[0:1].values
#temp.join(dataset1)
#merging dataset1 and dataset3 on the basis of id to add churn column

train_data = pd.merge(dataset1,dataset3,on="id")
X = train_data.iloc[:,1:37].values
Z = pd.DataFrame(X)
Y = train_data.iloc[:,37].values

#separating test data(the rows whose churn is not known)
a = dataset3.iloc[:,0].values
mask = dataset1['id'].isin(a)
test_data = dataset1[~mask]
pk = test_data
pk = pk['id']
pk = pd.DataFrame(pk)
test_data = test_data.iloc[:,1:].values


"""   TASK 3  :  finding relation between sales channel and churn  """

y = train_data.iloc[:,29].values
x = train_data.iloc[:,1].values
fig,axis=plt.subplots()
axis.scatter(x,y)
plt.title('sales channel and churn')
plt.xlabel('sales channel')
plt.ylabel('churn')
plt.show()

#splitting data into test and train set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


"""  TASK 4 : Building machine learning model predicting customers with high probability of churn   """


"""    MODEL 1  :     XGBoost model     """

# Fitting XGBoost to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_XGBoost = classifier.predict(test_data)

# Making the Confusion Matrix(number of right nad wrong predictions)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

#finding the accuracy of model
print(accuracies.mean()*100)

#finding the deviation in accuracy
print(accuracies.std()*100)

#merging prediction with respective ID of customers
Z = pd.DataFrame(y_pred_XGBoost)
pk = pd.concat([pk.reset_index(drop=True),Z.reset_index(drop=True)],axis=1)

#creating CSV file of predicted churn
pk.to_csv('Ravenclaw.csv', index=False)

"""    MODEL  2 :     ANN(artificial neural network) model      """

#Applying scaling to scale independent variables into a specific smaller range
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Creating the ANN(artificial neural network)
classifier = Sequential()

#input_layer
classifier.add(Dense(output_dim = 19, init = 'uniform' , activation = 'relu',input_dim = 36))

#hidden_layer
classifier.add(Dense(output_dim = 19, init = 'uniform' , activation = 'relu'))

#output layer
classifier.add(Dense(output_dim = 1, init = 'uniform' , activation = 'sigmoid'))

#compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

#fitting ANN to training set
classifier.fit(X_train, y_train, batch_size = 10, epochs =100)

"""   predicts the        PROBABILITY       of customer of leaving the bank   """

y_pred_prob = classifier.predict(X_test)
y_pred_ANN = classifier.predict(test_data)

#converting probability to true and false(0 and 1) for finding confusion matrix
p_pred = (y_pred_prob > 0.5)
p_pred_ANN = (y_pred_ANN > 0.5)

#making the confusion matrix(number of right nad wrong predictions)
cm = confusion_matrix(y_test, p_pred)
print(cm)

#finding the accuracy of model
accuracy= (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print(accuracy*100)

#improving the accuracy of ANN model
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 19, kernel_initializer = 'uniform', activation = 'relu', input_dim = 36))
    classifier.add(Dense(units = 19, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
y_pred_ANN = grid_search.predict(test_data)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_



"""    ANN is better since it also gives the probability of each customer of churning    """