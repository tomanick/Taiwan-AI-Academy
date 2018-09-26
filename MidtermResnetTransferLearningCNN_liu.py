
# coding: utf-8

# In[1]:


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


# In[2]:


# 資料在server上的路徑
path = '/data/examples/flower_classification'


# 將類別依據mapping.txt的分法用字典(命名為order)去歸類
order = {
    'daisy':0,
    'dandelion':1,
    'rose':2,
    'sunflower':3,
    'tulip':4
}


# In[3]:


# 將server路徑餵給os.listdir目錄並命名為dirs
dirs = os.listdir(path)
# dirs
# 執行後可知dirs下'依序'有這些子資料夾，其中train資料夾所在位移值為3


# In[4]:


dirs


# In[5]:


# load train data

y = []
# path+'/'+dirs[3]是train資料夾
# i in os.listdir(path+'/'+dirs[3])是train資料夾內的子資料夾
imglist = []
for i in os.listdir(path+'/'+dirs[3]):
# j in os.listdir(path+'/'+dirs[3]+'/'+i 是train資料夾內的子資料夾內(i)的檔案(圖片)
# 定義一個串列imglist去接收j的resize並跑一個迴圈
    for j in os.listdir(path+'/'+dirs[3]+'/'+i):
        img = cv2.imread(path+'/'+dirs[3]+'/'+i+'/'+j,3)
        img = cv2.resize(img, (224,224))
        img = img_to_array(img)
        imglist.append(img)
# 利用np.eye製造一個對角矩陣，對類別來做dummy index
        y.append(np.eye(5)[order[i]])


# In[6]:


train_data = np.array(imglist)


# In[7]:


train_data.shape


# In[8]:


y = np.array(y)


# In[9]:


#y_train = y


# In[10]:


os.listdir(path)


# In[11]:


len(os.listdir(path+'/'+dirs[2]))


# In[12]:


from keras.applications.resnet50 import preprocess_input, decode_predictions


# In[13]:


#Split training data into training and validation set.
#x_train = train_data
#x_data = train_data.astype('float32') - np.array([123.68, 116.78, 103.94])
x_data = preprocess_input(train_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.2, random_state=17, shuffle=True)


# In[14]:


# path+'/'+dirs[3]是train資料夾
# i in os.listdir(path+'/'+dirs[3])是test資料夾內的子資料夾
imglist_test = []
for i in os.listdir(path+'/'+dirs[2]):
# j in os.listdir(path+'/'+dirs[3]+'/'+i 是test資料夾內的子資料夾內(i)的檔案(圖片)
# 定義一個串列imglist去接收j的resize並跑一個迴圈
#     imglist = []
        img = cv2.imread(path+'/'+dirs[2]+'/'+i,3)
        img = cv2.resize(img, (224,224))
        img = img_to_array(img)
        #img.astype('float32')
        img = preprocess_input(img)
        imglist_test.append(img)


# In[15]:


test_data = np.array(imglist_test)


# In[16]:


test_data.shape


# In[17]:


#test_data[0]


# In[18]:


test_path = os.listdir(path+'/'+dirs[2])


# In[19]:


#test_path


# In[20]:


test_id = []
for i in test_path:
    test_1 = i.split('.')
    test_id.append(test_1[0])


# In[21]:


#test_id


# In[22]:


batch_size = 32
num_classes = 5
epochs = 100

save_dir = os.path.join(os.getcwd(), 'saved_models')


# In[23]:


import time
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import merge, Input
from keras.models import Model


# In[ ]:


##model##
from keras.layers import GlobalAveragePooling2D
img_size = 224

model_name = 'ResNet50-Fine-Tuning'

base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(img_size, img_size, 3))


#for layer in base_model.layers[-8:]:
    #layer.trainable=False


#for layer in base_model.layers:
    #layer.trainable = False

x = base_model.output
#l0 = Dropout(0.5)(x)
#l1 = GlobalAveragePooling2D()(x)
l1 = Flatten()(x)
l2 = Dense(1024, activation='relu')(l1)
l3 = Dense(2048, activation='relu')(l2)
#l4 = Dense(2048, activation='relu')(l3)
l4 = Dropout(0.5)(l3)
preds = Dense(num_classes, activation='softmax')(l3)
model = Model(inputs=base_model.input, outputs=preds)

#model.summary()


# In[ ]:


#model.summary()


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
#x_2 = []

img_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=0)


# In[ ]:


optimizer = keras.optimizers.Adam(lr=0.00001) 

#model_path = './saved_models/{}.h5'.format(model_name)

# Use ModelCheckpoint to save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    
model_path = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, 
                             verbose=1)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])


# earlystop
earlystop = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

# Fit the model on the batches generated by datagen.flow().
model_history = model.fit_generator(img_gen.flow(x_train, y_train,
                                 batch_size=batch_size),
                    epochs=50,
                    validation_data=(x_test, y_test),
                    workers=4,
                    callbacks=[earlystop, checkpoint])


# In[ ]:


#para = model.get_weights()


# In[ ]:


#para[0]


# In[ ]:


# loading our save model
print("Loading trained model")
model_best = load_model(model_path)
print("Loading has finished")


# In[ ]:


#checkpoint


# In[ ]:


prediction = model_best.predict(test_data).round()


# In[ ]:


pred_1 = np.argmax(prediction, axis=1)


# In[ ]:


submission = pd.DataFrame({'id': test_id, 'class': pred_1}, columns=['id', 'class'])
submission.to_csv('submission-random-17-100.csv',index=False)


# In[171]:


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[172]:


show_train_history(model_history, 'acc', 'val_acc')


# In[173]:


show_train_history(model_history, 'loss', 'val_loss')

