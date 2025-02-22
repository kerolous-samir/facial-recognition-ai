    #Import libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import copy
    from sklearn.model_selection import train_test_split
    from tensorflow import keras
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.models import Model, load_model, Sequential
    from tensorflow.keras.initializers import glorot_uniform
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
    from IPython.display import display
    from tensorflow.python.keras import *
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import layers, optimizers
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.layers import *
    from tensorflow.keras import backend as K
    from keras import optimizers
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import cv2
    import tensorflow as tf
    import json 
    import tensorflow.keras.backend as K
    import os

kf = pd.read_csv("Emotion AI Dataset\\data.csv") 
kf['Image'] = kf['Image'].apply(lambda x: np.fromstring(x, dtype= int, sep=" ").reshape(96,96))

i = np.random.randint(1,len(kf))
plt.imshow(kf['Image'][i],cmap='gray')
for j in range(1,31,2):
    plt.plot(kf.loc[i][j-1],kf.loc[i][j],"rx")

import random
fig = plt.figure(figsize=(20,20))
for i in range(64):
    k = random.randint(0,len(kf)+1)
    ax = fig.add_subplot(8,8,i+1)
    image = plt.imshow(kf['Image'][i],cmap='gray')
    for j in range(1,31,2):
        plt.plot(kf.loc[i][j-1],kf.loc[i][j],"rx")


#preform image augmentation
kf_cp = copy.copy(kf)
columns = kf.columns[:-1]
kf_cp["Image"] = kf_cp["Image"].apply(lambda x: np.flip(x,axis=1))

for i in range(len(columns)):
    if i%2 == 0:
        kf_cp[columns[i]] = kf_cp[columns[i]].apply(lambda x: 96. - float(x))

plt.imshow(kf['Image'][5],cmap='gray')
for j in range(1,31,2):
    plt.plot(kf.loc[5][j-1],kf.loc[5][j],"rx")


plt.imshow(kf_cp['Image'][5],cmap='gray')
for j in range(1,31,2):
    plt.plot(kf_cp.loc[5][j-1],kf_cp.loc[5][j],"rx")

augmanted_df = np.concatenate((kf,kf_cp))

kf_copy = copy.copy(kf)
kf_copy["Image"] = kf_copy["Image"].apply(lambda x: np.clip(random.uniform(1.5,2.)*x,0.,255.))

augmanted_df = np.concatenate((augmanted_df,kf_copy))

kf_flip_x = copy.copy(kf)

kf_flip_x["Image"] = kf_flip_x["Image"].apply(lambda x: np.flip(x,axis=0))
plt.imshow(kf_flip_x['Image'][5],cmap='gray')


for i in range(len(columns)):
    if i%2 != 0:
        kf_flip_x[columns[i]] = kf_flip_x[columns[i]].apply(lambda x: 96. -float(x))
    


plt.imshow(kf_flip_x['Image'][5],cmap='gray')
for j in range(1,31,2):
    plt.plot(kf_flip_x.loc[5][j-1],kf_flip_x.loc[5][j],"rx")

img = augmanted_df[:,30]
img = img/255.
img.shape

for i in range(len(img)):
    X[i,] = np.expand_dims(img[i],axis=2)

X = np.asarray(X).astype(np.float32)

y = augmanted_df[:,:30]
y = np.asarray(y).astype(np.float32)

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)


# Import necessary libraries
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense, Dropout, ZeroPadding2D
from tensorflow.keras.callbacks import ModelCheckpoint

# Assuming you have loaded and preprocessed your data (kf, kf_cp, kf_copy, kf_flip_x, X_train, y_train, etc.)

# Define the function for residual block
def res_block(X,filter,stage):
    X_copy = X
    f1 , f2 , f3 = filter
    
    #main path
    X = Conv2D(f1,(1,1),strides=(1,1),name= "res_"+str(stage)+"_conv_a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = MaxPool2D()(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_conv_a")(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f2,(3,3),strides=(1,1),name= "res_"+str(stage)+"_conv_b",padding = 'same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_conv_b")(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f3,(1,1),strides=(1,1),name= "res_"+str(stage)+"_conv_c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_conv_c")(X)
    
    #short path
    X_copy = Conv2D(f3,(1,1),strides=(1,1),name= "res_"+str(stage)+"_conv_copy", kernel_initializer=glorot_uniform(seed=0))(X_copy)
    X_copy = MaxPool2D()(X_copy)
    X_copy = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_conv_copy")(X_copy)
    
    #adding
    X = Add()([X,X_copy])
    X = Activation('relu')(X)
    
    #Identity block 1
    X_copy = X
    X = Conv2D(f1,(1,1),strides=(1,1),name= "res_"+str(stage)+"_id_1_a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_id_1_a")(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f2,(3,3),strides=(1,1),name= "res_"+str(stage)+"_id_1_b",padding = 'same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_id_1_b")(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f3,(1,1),strides=(1,1),name= "res_"+str(stage)+"_id_1_c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_id_1_c")(X)
    
    #adding
    X = Add()([X,X_copy])
    X = Activation('relu')(X)
    
    
    #Identity block 2
    X_copy = X
    X = Conv2D(f1,(1,1),strides=(1,1),name= "res_"+str(stage)+"_id_2_a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_id_2_a")(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f2,(3,3),strides=(1,1),name= "res_"+str(stage)+"_id_2_b",padding = 'same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_id_2_b")(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f3,(1,1),strides=(1,1),name= "res_"+str(stage)+"_id_2_c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_id_2_c")(X)
    
    #adding
    X = Add()([X,X_copy])
    X = Activation('relu')(X)
    
    return X

# Define the model architecture
def create_model():
    X_input = Input((96, 96, 1))
    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name="conv_1", kernel_initializer='glorot_uniform')(X)
    X = BatchNormalization(axis=3, name="bn_conv_1")(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage-2
    X = res_block(X, filter=[64, 64, 256], stage=2)

    # Stage-3
    X = res_block(X, filter=[128, 128, 512], stage=3)

    # Stage-4 (commented out, you can add if needed)
    #X = res_block(X, filter=[256, 256, 1024], stage=4)

    # Average pooling    
    X = AveragePooling2D(pool_size=(2, 2), name="average_pooling")(X)

    X = Flatten()(X)
    X = Dense(4096, activation='relu')(X)
    X = Dropout(0.2)(X)
    X = Dense(2048, activation='relu')(X)
    X = Dropout(0.1)(X)
    X = Dense(30, activation='relu')(X)

    # Final model
    model = Model(inputs=X_input, outputs=X)
    return model

# Create the model instance
model_key_facial = create_model()

# Compile the model
model_key_facial.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Print model summary
model_key_facial.summary()

# Model checkpoint to save the best model
checkpointer = ModelCheckpoint(filepath="facialkey_points.keras", verbose=1, save_best_only=True)

# Train the model on GPU
with tf.device('/GPU:1'):  # Adjust '/GPU:0' according to your GPU index
    history = model_key_facial.fit(X_train, y_train, batch_size=16, epochs=1, validation_split=0.05, callbacks=[checkpointer])

# Save model architecture as JSON
model_json = model_key_facial.to_json()
with open("facial.json", "w") as json_file:
    json_file.write(model_json)


model_key_facial.save_weights(".weights.h5")
with open("facial.json","r") as json_model:
    best_model = json_model.read()
model_1_key_facial = tf.keras.models.model_from_json(best_model)
model_1_key_facial.load_weights("Emotion AI Dataset\\weights.hdf5")
model_1_key_facial.compile(optimizer="adam",loss="mean_squared_error",metrics=["accuracy"])
model_1_key_facial.evaluate(X_test,y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.legend(["loss","val_loss"],loc="upper right")
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()

facial_exp = pd.read_csv("Emotion AI Dataset\\icml_face_data.csv")

def img2array(x):
    return np.array(x.split(" ")).reshape(48,48,1).astype('float32')
def changedim(x):
    img = x.reshape(48,48)
    return cv2.resize(img,dsize=(96,96),interpolation= cv2.INTER_CUBIC)
facial_exp[' pixels'] = facial_exp[' pixels'].apply(lambda x: img2array(x))

facial_exp[' pixels'] = facial_exp[' pixels'].apply(lambda x: changedim(x))

labeltotext = {0: 'anger', 1: 'disgust', 2: 'sad', 3: 'happiness', 4: 'surprise'}
plt.imshow(facial_exp[' pixels'][5],cmap='grey')

emotions = list(range(0,5))
for i in emotions:
    data = facial_exp[facial_exp['emotion'] == i][:1]
    img = data[" pixels"].item()
    plt.figure()
    plt.title(labeltotext[i])
    plt.imshow(img,cmap='grey')
    

plt.figure(figsize=(12,12))
sns.barplot(x=facial_exp['emotion'].value_counts().index,y=facial_exp['emotion'].value_counts())

from keras.utils import to_categorical
X = facial_exp[' pixels']
y = to_categorical(facial_exp['emotion'])

X = np.stack(X,axis=0)

X = X.reshape(24568, 96, 96,1)

X_train , X_Test , y_train , y_Test = train_test_split(X,y,test_size=0.1,shuffle=True)


X_Test , X_val , y_Test , y_val = train_test_split(X_Test,y_Test,test_size=0.5, shuffle=True)

X_train = X_train/255
X_Test = X_Test/255
X_val = X_val/255
img_aug = ImageDataGenerator(rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,zoom_range=0.1,
                            shear_range=0.1,fill_mode='nearest',brightness_range=[1.1,1.5],vertical_flip=True)


# Import necessary libraries
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense, Dropout, ZeroPadding2D
from tensorflow.keras.callbacks import ModelCheckpoint

def res_block(X,filter,stage):
    X_copy = X
    f1 , f2 , f3 = filter
    
    #main path
    X = Conv2D(f1,(1,1),strides=(1,1),name= "res_"+str(stage)+"_conv_a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = MaxPool2D()(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_conv_a")(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f2,(3,3),strides=(1,1),name= "res_"+str(stage)+"_conv_b",padding = 'same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_conv_b")(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f3,(1,1),strides=(1,1),name= "res_"+str(stage)+"_conv_c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_conv_c")(X)
    
    #short path
    X_copy = Conv2D(f3,(1,1),strides=(1,1),name= "res_"+str(stage)+"_conv_copy", kernel_initializer=glorot_uniform(seed=0))(X_copy)
    X_copy = MaxPool2D()(X_copy)
    X_copy = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_conv_copy")(X_copy)
    
    #adding
    X = Add()([X,X_copy])
    X = Activation('relu')(X)
    
    #Identity block 1
    X_copy = X
    X = Conv2D(f1,(1,1),strides=(1,1),name= "res_"+str(stage)+"_id_1_a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_id_1_a")(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f2,(3,3),strides=(1,1),name= "res_"+str(stage)+"_id_1_b",padding = 'same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_id_1_b")(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f3,(1,1),strides=(1,1),name= "res_"+str(stage)+"_id_1_c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_id_1_c")(X)
    
    #adding
    X = Add()([X,X_copy])
    X = Activation('relu')(X)
    
    
    #Identity block 2
    X_copy = X
    X = Conv2D(f1,(1,1),strides=(1,1),name= "res_"+str(stage)+"_id_2_a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_id_2_a")(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f2,(3,3),strides=(1,1),name= "res_"+str(stage)+"_id_2_b",padding = 'same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_id_2_b")(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f3,(1,1),strides=(1,1),name= "res_"+str(stage)+"_id_2_c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= "bn_"+str(stage)+"_id_2_c")(X)
    
    #adding
    X = Add()([X,X_copy])
    X = Activation('relu')(X)
    
    return X

# Define the model architecture
def create_model():
    X_input = Input((96, 96, 1))
    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name="conv_1", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name="bn_conv_1")(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage-2
    X = res_block(X, filter=[64, 64, 256], stage=2)

    # Stage-3
    X = res_block(X, filter=[128, 128, 512], stage=3)

    # Stage-4 (commented out, you can add if needed)
    #X = res_block(X, filter=[256, 256, 1024], stage=4)

    # Average pooling    
    X = AveragePooling2D(pool_size=(4, 4), name="average_pooling")(X)

    X = Flatten()(X)
    X = Dense(5, activation='softmax',kernel_initializer= glorot_uniform(seed=0),name='dense_final')(X)
    # Final model
    model = Model(inputs=X_input, outputs=X)
    return model

# Create the model instance
model_2_emotion = create_model()

# Compile the model
model_2_emotion.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model_2_emotion.summary()

early_stopping = EarlyStopping(monitor='val_loss', mode='min',patience=10,verbose=1)

# Model checkpoint to save the best model
checkpointer = ModelCheckpoint(filepath="emotion_points.keras", verbose=1, save_best_only=True)

# Train the model on GPU
with tf.device('/GPU:1'):  # Adjust '/GPU:0' according to your GPU index
    history = model_2_emotion.fit(img_aug.flow(X_train, y_train,batch_size=64) ,validation_data=(X_val,y_val) ,
                                  epochs=10, steps_per_epoch=len(X_train) // 64 , callbacks=[checkpointer,early_stopping])

# Save model architecture as JSON
model_json = model_2_emotion.to_json()
with open("emotion.json", "w") as json_file:
    json_file.write(model_json)



with open("emotion.json","r") as json_model:
    emotion_model = json_model.read()
model_1_emotion = tf.keras.models.model_from_json(emotion_model)
model_1_emotion.load_weights("Emotion AI Dataset\\weights_emotions.hdf5")

model_1_emotion.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

score = model_1_emotion.evaluate(X_Test,y_Test)

np.argmax(model_1_emotion.predict(X_Test),axis=-1).shape

model_1_emotion.predict(X_Test)
predict_classes = np.argmax(model_1_emotion.predict(X_Test),axis=-1)
y_true = np.argmax(y_Test,axis=-1)
from sklearn.metrics import confusion_matrix , precision_score , classification_report
cm = confusion_matrix(y_true,predict_classes)
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True,cbar=False)
ps = classification_report(y_true,predict_classes)

L = 5
W = 5
fig , axes = plt.subplots(L,W,figsize = (10,10))
axes = axes.ravel()
for i in np.arange(0,L*W):
    axes[i].imshow(X_Test[i].reshape(96,96),cmap = 'grey')
    axes[i].set_title(f'Predection = {labeltotext[predict_classes[i]]}\n True = {labeltotext[y_true[i]]}')
    axes[i].axis('off')
    
plt.subplots_adjust(wspace=1)
def predict(X_test):
    
    df_key_facial = model_1_key_facial.predict(X_test)
    
    df_emotion = np.argmax(model_1_emotion.predict(X_test),axis=-1)
    
    df_emotion = np.expand_dims(df_emotion,axis=1)
    
    df_model = pd.DataFrame(df_key_facial)
    
    df_model['emotions'] = df_emotion
    
    return df_model
    

df_model = predict(gray_images)

model_1_emotion.predict(gray_images)

L = 4
W = 4
fig , axes = plt.subplots(L,W,figsize = (16,16))
axes = axes.ravel()
for i in range(16):
    axes[i].imshow(X_test[i].squeeze(),cmap = 'grey')
    axes[i].set_title(f"Predection = {labeltotext[df_model['emotions'][i]]}")
    axes[i].axis('off')
    
    for j in range(1,31,2):
        axes[i].plot(df_model.loc[i][j-1],df_model.loc[i][j],"rx")
plt.subplots_adjust(wspace=1)
