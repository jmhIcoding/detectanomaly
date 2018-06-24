__author__ = 'jmh081701'
# The basic file support sequence predict.
import os
import  sys
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import  tensorflow as tf

class Predictor:
    def __init__(self,
                 modelname,
                 feature_n=1,
                 windows_length=3,
                 predict_length=1,
                 hidden_layers=[10,20]
    ):
        '''
            :param modelname: a name to describe this predictor.
            :param feature_n: the feature number of each element ,that is the rnn network 's frame_size.
                    For example:
                    input vector: [[1,2],[2,3],.....] then  the feature number is two.
            :param windows_length: The look back size used when scanning the input vector.
            :param predict_length: The predict length
            :param hidden_layers:  The hidden layer units. This parameter should be a list of int,
                    For example:hidden_layers=[20,30],indicts that there are two lstm layer stacking together,
                    and the first lstm layer has 20 hidden units,where the second layer has 30 hidden units.
            :return: self
        '''
        self.model_file=modelname+".mol"
        self.parameterfile =modelname+".para"
        self._model =None
        self.feature_n=feature_n
        self.hidden_layers=hidden_layers
        self.predict_length =predict_length
        self.windows_length =windows_length
        self._min=[math.inf for i in range(self.feature_n)]
        self._max=[-math.inf for i in range(self.feature_n)]
        if os.path.exists(os.path.realpath(self.model_file)) and os.path.exists(os.path.realpath(self.parameterfile)):
            #load model from history file.
            print("load model from %s and %s"%(self.model_file,self.parameterfile))
            self.load_model(self.model_file,self.parameterfile)
        else:
            #build new model
            self._model = Sequential()
            if len(self.hidden_layers)==1:
                self._model.add(LSTM(self.hidden_layers[0],input_shape=(self.windows_length,self.feature_n),return_sequences=False))
                self._model.add(Dense(self.predict_length*self.feature_n))
            else:
                self._model.add(LSTM(self.hidden_layers[0],input_shape=(self.windows_length,self.feature_n),return_sequences=True))
                for index in range(1,len(self.hidden_layers)-1):
                    self._model.add(LSTM(self.hidden_layers[index],return_sequences=True))
                self._model.add(LSTM(self.hidden_layers[-1]))
                self._model.add(Dense(self.feature_n*self.predict_length))
                self._model.compile(loss="mean_squared_error",optimizer="adam")

    def load_model(self,modelfile=None,parameterfile=None):
        '''
        #load model from history file.
            :param modelfile: modelfile is the weight parameter file.
            :param parameterfile: parameterfile should be the super parameter file,such as featurn_n ...
            :return:
        '''
        if modelfile==None:
            modelfile = self.model_file
        if parameterfile == None:
            parameterfile =self.parameterfile

        #load Sequence mode .
        self._model=load_model(modelfile)

        # load parameter from json file.
        with open(parameterfile,"r") as fp :
            para=json.load(fp)
            self.feature_n =para["feature_n"]
            self.predict_length =para["predict_length"]
            self.hidden_layers =para["hidden_layers"]
            self.windows_length=para["windows_length"]
            self._min = para["_min"]
            self._max = para["_max"]

    def save_model(self,modelfile=None,parameterfile=None):
        if modelfile ==None :
            modelfile = self.model_file
        if None == parameterfile:
            parameterfile = self.parameterfile

        # save sequence model,
        self._model.save(modelfile)

        # save parameter to json.
        with open(parameterfile,"w") as fp:
            json.dump({"feature_n":self.feature_n,
                       "predict_length":self.predict_length,
                       "hidden_layers":self.hidden_layers,
                       "windows_length":self.windows_length,
                       "_min":self._min,
                       "_max":self._max}
                      ,fp)
    def max_min_transform(self,_x):
        #max-min scalar
        #convert a vector to 0-1,with the method (x[?][i]-min)/(max-min)
        x = _x.copy()
        for each in x:
            for i in range(self.feature_n):
                self._min[i] = min(each[i]/2,self._min[i])
                self._max[i] = max(each[i]*2,self._max[i])
        for j in range(len(x)):
            for i in range(self.feature_n):
                x[j][i] =(x[j][i]-self._min[i])/(self._max[i]-self._min[i]+0.000000001)
        return x
    def max_min_inverse_transform(self,_x):
        #max-min scalar
        #inverse convert a vector to min -max ,with the method (x[?][i]-min)/(max-min)
        # the shape should be :(?,featurn_n)
        x = _x.copy()
        for j in range(len(x)):
            for i in range(self.feature_n):
                x[j][i] =x[j][i]*(self._max[i]-self._min[i]) + self._min[i]
        return x
    def predict(self,x,y=None):
        #given self.windows_length elems to predict next sequence
        # The Input Should be non-convert to 0-1
        '''
            :param x: given self.windows_length elems to predict next sequence.
            :param y: reserved
            :return:
        '''
        for i in range(len(x)):
            x[i] = self.max_min_transform(x[i])
        x = np.array(x)
        x = np.reshape( x,newshape=(x.shape[0],x.shape[1],self.feature_n))
        predicty = self._model.predict(x)
        #print(predicty)
        return predicty
    def predict_once(self,x):
        # predict only once ,that's : x should only include self.windows_length elem.
        '''
            For example :
                windows_length :3
                feature_n: 2
                then the shape of x should be : (3,2)
                the output's shape will be (predict_length,featurn_n)
        '''
        x = self.max_min_transform(x)
        x = np.array([x])
        x = np.reshape( x,newshape=(x.shape[0],x.shape[1],self.feature_n))
        predicty = self._model.predict(x)
        y = self.max_min_inverse_transform(predicty)[0]
        return y

    def _gen_data(self,x):
        # according the input x vector,and the parameter windows_length,predict_length  to generate the pairs for
        #train or predict
        '''
            For examples:
                x:[[1],[2],[3],[5],[9],[8],[10],[23]]
                windows_length:3
                predict_length:2
                featurn_n   : 1
            Then the output should be:
                ret_x:[[[1],[2],[3]],[[2],[3],[5]],[[3],[5],[9]],[[5],[9],[8]]]
                ret_y:[ [[5],[9]],  [[9],[8]] ,[[8],[10]],[[10],[23]]]
                because: [[5],[9]] is the later two element behind [[1],[2],[3]] and so on.
            While this is another examples:
                x:[[0, 0], [1, 2], [2, 4], [3, 6], [4, 8], [5, 10]]
                windows_length:3
                predict_length:1
                feature_n   : 2
            Then the output should be:
                ret_x : [ [ [0,0],[1,2],[2,4] ],[ [1,2],[2,4],[3,6] ],[ [2,4],[3,6],[4,8] ]]
                ret_y : [ [ [3,6] ],                [ [4,8] ],              [ [5,10] ] ]
        '''
        ret_x=[]
        ret_y=[]
        # check the shape
        if len(x[0]) != self.feature_n:
            s="Error input x vector,for each elem in x expect (%d),but get (%d)"%(self.feature_n,len(x[0]))
            raise s
        for index  in range(0,len(x)-self.predict_length-self.windows_length):
            ret_x.append(x[index:(index+self.windows_length)])
            ret_y.append(x[(index+self.windows_length):(index+self.windows_length+self.predict_length)])

        return np.array(ret_x),np.array(ret_y)

    def train(self,x,y,epochs=400,batch_size=1,verbose=2):
        x = np.reshape(x,newshape=(x.shape[0],x.shape[1],self.feature_n))
        y = np.reshape(y,newshape=(-1,self.predict_length*self.feature_n))
        self._model.fit(x,y,batch_size,epochs,verbose)
        self.save_model(self.model_file,self.parameterfile)

tool =Predictor(modelname="test",feature_n=2,predict_length=1)
print("*"*50)
x=[]
for i in range(0,200):
    x.append([i,2*i])
print(x)
x = tool.max_min_transform(x)
print(x)
trainX,trainY=tool._gen_data(x)
tool.train(trainX,trainY,epochs=200)
y =tool.predict_once( [[198, 396], [199, 398], [200, 400]])
print(y)
#print(tool.predict_once( [[300, 396], [199, 398], [200, 400]]))
print("#"*40)

