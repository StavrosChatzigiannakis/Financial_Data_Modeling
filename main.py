import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_info = pd.read_csv(r'C:\Users\stayr\PycharmProjects\Lending_Club_Data\lending_club_info.csv',index_col='LoanStatNew')
df = pd.read_csv(r'C:\Users\stayr\PycharmProjects\Lending_Club_Data\lending_club_loan_two.csv')
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})

#Data Preprocessing
 #Missing Data
df = df.drop('emp_title',axis=1)
df = df.drop('emp_length',axis=1)
df = df.drop('title',axis=1)

 #Filling in the missing mort_acc values
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']

def fill_mort_acc(total_acc, mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

 #Dropping remaining N/A Values (<0,5%)
df = df.dropna()

 #Formatting String Objects
df['term'] = df['term'].apply(lambda term: int(term[:3]))

 #grade feature
df = df.drop('grade',axis=1)
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)

 #verification_status, application_type,initial_list_status,purpose
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)

 #home_ownership
df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)

 #address
df['zip_code'] = df['address'].apply(lambda address:address[-5:])
dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)

 #issue_d
df = df.drop('issue_d',axis=1)

 #earliest_cr_line
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)

#Train test split
from sklearn.model_selection import train_test_split
 #dropping duplicates
df = df.drop('loan_status',axis=1)

X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=96)

#Data Normalizing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Creating the model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm

model = Sequential()

# input layer
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1,activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train,
          y=y_train,
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test),
          )

#SAVING THE MODEL
from tensorflow.keras.models import load_model
model.save('full_data_project_model.h5')
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
plt.show()

from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))