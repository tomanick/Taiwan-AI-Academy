
# coding: utf-8

# In[130]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image, ImageOps


import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import img_to_array, load_img

import matplotlib.pyplot as plt
import os


# In[131]:


# 資料在server上的路徑
path = '/data/examples/may_the_4_be_with_u/where_am_i'
# 將類別依據mapping.txt的分法用字典(命名為order)去歸類
order = {'CALsuburb': 9,
'PARoffice': 7,
'bedroom': 12,
'coast': 10,
'forest': 4,
'highway': 14,
'industrial':2,
'insidecity': 3,
'kitchen': 0,
'livingroom': 5,
'mountain': 8,
'opencountry': 6,
'store': 11,
'street': 1,
'tallbuilding': 13}


# In[132]:


# 將server路徑餵給os.listdir目錄並命名為dirs
dirs = os.listdir(path)
# dirs
# 執行後可知dirs下'依序'有這些子資料夾，其中train資料夾所在位移值為3


# In[133]:


y = []
# path+'/'+dirs[3]是train資料夾
# i in os.listdir(path+'/'+dirs[3])是train資料夾內的子資料夾
imglist = []
for i in os.listdir(path+'/'+dirs[3]):
# j in os.listdir(path+'/'+dirs[3]+'/'+i 是train資料夾內的子資料夾內(i)的檔案(圖片)
# 定義一個串列imglist去接收j的resize並跑一個迴圈
    for j in os.listdir(path+'/'+dirs[3]+'/'+i):
        img = cv2.imread(path+'/'+dirs[3]+'/'+i+'/'+j,3)
        img = cv2.resize(img, (150,150))
        img = img_to_array(img)
        imglist.append(img)
# 利用np.eye製造一個對角矩陣，對類別來做dummy index
        y.append(np.eye(15)[order[i]])


# In[134]:


train_data = np.array(imglist)


# In[135]:


train_data.shape


# In[136]:


y = np.array(y)


# In[137]:


#y_train = y


# In[138]:


os.listdir(path)


# In[139]:


len(os.listdir(path+'/'+dirs[2]))


# In[140]:


#Split training data into training and validation set.
x = train_data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=11, shuffle=True)


# In[141]:


x_train.shape


# In[142]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.


# In[143]:


# path+'/'+dirs[3]是train資料夾
# i in os.listdir(path+'/'+dirs[3])是test資料夾內的子資料夾
imglist_test = []
for i in os.listdir(path+'/'+dirs[2]):
# j in os.listdir(path+'/'+dirs[3]+'/'+i 是test資料夾內的子資料夾內(i)的檔案(圖片)
# 定義一個串列imglist去接收j的resize並跑一個迴圈
#     imglist = []
        img = cv2.imread(path+'/'+dirs[2]+'/'+i,3)
        img = cv2.resize(img, (150,150))
        img = img_to_array(img)
        img.astype('float32')
        img /= 255
        imglist_test.append(img)


# In[144]:


test_data = np.array(imglist_test)


# In[145]:


test_data.shape


# In[146]:


test_path = os.listdir(path+'/'+dirs[2])


# In[147]:


test_id = []
for i in test_path:
    test_1 = i.split('.')
    test_id.append(test_1[0])


# In[149]:


batch_size = 64
num_classes = 15
epochs = 55
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'CNN_v2.1'


# In[150]:


# build our CNN model
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',
                  input_shape=x_train.shape[1:], activation='relu'))                 
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same',
                  activation='relu'))   
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same',
                  activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same',
                  activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same',
                  activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())

# initiate Adam optimizer
opt = keras.optimizers.Adam()

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


print('Using real-time data augmentation.')

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# Use ModelCheckpoint to save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)

# earlystop
earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Fit the model on the batches generated by datagen.flow().
model_history = model.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    workers=4,
                    callbacks=[earlystop])

# loading our save model
print("Loading trained model")
model = load_model(model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[108]:


prediction = model.predict_classes(test_data)
submission = pd.DataFrame({'id': test_id, 'class': prediction}, columns=['id', 'class'])
submission.to_csv('submission.csv',index=False)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script ResnetTransferLearningCNN_liu.ipynb')

