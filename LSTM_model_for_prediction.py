import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import math

#Load the LSTM model
new_model=tf.keras.models.load_model('lstm_model_sine_wave_using_csv_file.h5')

#load csv file
new_df=pd.read_csv('LSTM_sine_data.csv')
window=10

#generate random number to test LSTM model
xtest=[]
for i in range(0,100,1):
    n=round(random.random(),2)
    n2=math.sin(i*np.pi/30)+n*0.3
    xtest.append((n2))
    
plt.figure()
plt.plot(xtest)
plt.title(label='Testing data', color = 'blue', fontsize='20')

#print('\n 1. xtest after appending noise: ', xtest)

#store window point as a sequence    
xin=[]

for i in range(window,len(xtest)):
    xin.append(xtest[i-window:i])

#print('\n 2. xin after windowing: ', xin)   

#reshaping data to format for LSTM
xin = np.array(xin)
#print('\n 3. xin after formatting into numpy array: ', xin)

xin=xin.reshape(xin.shape[0],xin.shape[1],1)
#print('\n 4. xin after reshaping into 3D LSTM format: ', xin)



#predict data in format for LSTM
x_pred=new_model.predict(xin)
#print('\n 1st prediction: ', x_pred)

#plot prediction 
plt.figure()
plt.plot(x_pred)
plt.title(label='1st prediction plot', color = 'red', fontsize = '20')



xtest=np.array(xtest)#mark

#using predicted value to presict next step
x_pred=xtest.copy()#mark


#print('\n x_pred: ', x_pred)
for i in range(window,len(xtest)):
    xin= x_pred[i-window:i].reshape((1, window, 1))
    x_pred[i]=new_model.predict(xin)
    

plt.figure()
plt.plot(x_pred)
plt.title(label='2nd prediction plot', color = 'green', fontsize = '20')

plt.show()

