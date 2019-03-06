import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C:/Users/ashis/OneDrive/Documents/deep_learning_a2z/ann/Artificial_Neural_Networks/Churn_Modelling.csv")

x = dataset.iloc[:,3:13].values # for all the independent variables
y = dataset.iloc[:,13].values # final output predictions

# the input variables are in the object form i.e because of the presence of categorical variables so our task is to convert
#the categorical variables into numerical or the ordinal variables
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
#since 2 categories are present

lb_en_x1 = LabelEncoder()
x[:,1] = lb_en_x1.fit_transform(x[:,1])

lb_en_x2 = LabelEncoder()
x[:,2] = lb_en_x1.fit_transform(x[:,2])

one_hot_encoder = OneHotEncoder(categorical_features= [1])
x = one_hot_encoder.fit_transform(x).toarray()

#to remove dummy variable trap
x = x[:,1:]

# splitting the dataset into test set and training set
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#now applying feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#building the neural network
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialzing the model

classifier = Sequential()

classifier.add(Dense(output_dim = 6,init = 'uniform',activation='relu',input_dim= 11))
#single hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform',activation='relu'))

#adding the output layer
classifier.add(Dense(output_dim = 1,init = 'uniform',activation='sigmoid'))

#compiling the NN
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])


#fitting the NN
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

#predicting the test values
y_pred = classifier.predict(x_test)

y_pred = (y_pred >0.5)

#make the confusionmatrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

accuracy = (1548 + 134)/2000

"""home work
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))

new_prediction = (new_prediction >0.5)








