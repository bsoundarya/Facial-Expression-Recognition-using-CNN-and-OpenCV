#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import tensorflow 
import keras
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense,MaxPooling2D, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D

from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.optimizers import *


#---------------------------------------------------------------------------------------------------------------------------------
def generate_dataset():
    
    """generate dataset from csv"""
    
    df = pd.read_csv(r"C:\Users\sbabu5\Downloads\fer2013\fer2013.csv")
    
    train_samples = df[df['Usage']=="Training"]
    validation_samples = df[df["Usage"]=="PublicTest"]
    test_samples = df[df["Usage"]=="PrivateTest"]
    
    y_train = train_samples.emotion.astype(np.int32).values
    y_valid = validation_samples.emotion.astype(np.int32).values
    y_test = test_samples.emotion.astype(np.int32).values
     
    X_train =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape((48,48)) for image in train_samples.pixels])
    X_valid =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape((48,48)) for image in validation_samples.pixels])
    X_test =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape((48,48)) for image in test_samples.pixels])
    
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

#---------------------------------------------------------------------------------------------------------------------------------
    
def generate_model(lr=0.001):
    
    
    """training model"""
    
    with tf.device('/gpu:0'):
        model=Sequential()
        # 1 - Convolution
        model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.20))


        # Flattening
        model.add(Flatten())

        # Fully connected layer 1st layer
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.20))


        # Fully connected layer 2nd layer
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.20))

        model.add(Dense(10, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        #fitting the model()
        
       
        return model
    
#---------------------------------------------------------------------------------------------------------------------------------
        
if __name__=="__main__":
    
    #df = pd.read_csv("./fer2013/fer2013.csv")
    X_train, y_train, X_valid, y_valid, X_test, y_test =  generate_dataset()
    
    X_train = X_train.reshape((-1,48,48,1)).astype(np.float32)
    X_valid = X_valid.reshape((-1,48,48,1)).astype(np.float32)
    X_test = X_test.reshape((-1,48,48,1)).astype(np.float32)
    
    X_train_std = X_train/255.
    X_valid_std = X_valid/255.
    X_test_std = X_test/255.
    
    model = generate_model(0.01)
    with tf.device("/gpu:0"):
        history = model.fit(X_train_std, y_train,batch_size=128,epochs=40, validation_data=(X_valid_std, y_valid), shuffle=True)
        #model.save(r"C:\Users\sbabu5\Downloads\my_model.h5")
        save_model = 'model11'
        model_loc = os.path.join(save_model,'model11.h5')
    
    
    
    


# In[5]:


from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt
import scikitplot
import seaborn as sns


# In[6]:


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[7]:


from sklearn.metrics import classification_report, confusion_matrix


pred_list = []; actual_list = []

predictions = model.predict(X_test)

for i in predictions:
    pred_list.append(np.argmax(i))

for i in y_test:
    actual_list.append(np.argmax(i))

confusion_matrix(actual_list, pred_list)

scikitplot.metrics.plot_confusion_matrix(pred_list,y_test,figsize=(7,7))
plt.show()

print(classification_report(y_test,pred_list))
print('Accuracy:',accuracy_score(y_test,pred_list))


# In[8]:


model.summary()


# In[9]:


score = model.evaluate(X_valid, y_valid, verbose=1) 
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)


# In[10]:


""" metrics collected by history object """
history_dict=history.history
history_dict.keys()


# In[11]:


print(history_dict["accuracy"])


# In[12]:


import matplotlib.pyplot as plt

train_loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, train_loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[13]:


model_json = model.to_json()
model.save_weights('model_weights.h5')
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# In[14]:


from tensorflow.keras.models import model_from_json
class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]
    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()
    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]


# In[16]:


import cv2
facec = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
        return fr


# In[17]:


def gen(camera):
    while True:
        frame = camera.get_frame()
        cv2.imshow('Facial Expression Recognization',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


# In[ ]:


gen(VideoCamera())


# In[ ]:




