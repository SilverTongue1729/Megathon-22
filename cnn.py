def mean(l):
    return sum(l)/len(l)

import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#for loading and visualizing audio files
import librosa
import librosa.display
import numpy as np
#to play audio
import IPython.display as ipd
import tensorflow as tf

# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, Flatten
# from keras.utils import to_categorical
accent_fpath = "/home/roja_26/Year2/megathon/Megathon-Proj/train_wav"
accent_clips = os.listdir(accent_fpath)
max_audiosig=2076
X_train=np.array([])
X_test=np.array([])
y_train=np.array([i//2 for i in range(0,20)])
y_test=np.array([i for i in range(0,10)])
print(y_test)
# y_train = y_train.reshape(20)
# y_train = tf.convert_to_tensor(y_train)
y_train=tf.keras.utils.to_categorical(y_train)
y_test=tf.keras.utils.to_categorical(y_test)
print(y_test)


for i in accent_clips:
    audio_fpath = "/home/roja_26/Year2/megathon/Megathon-Proj/train_wav/"+i
    audio_clips = os.listdir(audio_fpath)
    if(len(audio_clips)>=29):
        # lang=[0]*10
        # lang[c]=1
        c=0
        for j in audio_clips[0:3]:
            # y_train=np.append(y_train,lang)
            clip_fpath = audio_fpath+"/"+j
            # MFCC
            # plt.figure(figsize=(14, 5))
            y, sr = librosa.load(clip_fpath, sr=22050)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            if (max_audiosig > mfcc.shape[1]):
                pad_width = max_audiosig - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            if(c<2):
                X_train=np.append(X_train,mfcc)
            else:
                X_test=np.append(X_test,mfcc)
            c+=1

print(X_train.shape)
X_train = X_train.reshape(20,20,2076,1)
X_test = X_test.reshape(10,20,2076,1)
# y_train = y_train.reshape(20)
# y_train = tf.keras.utils.to_categorical(y_train)


def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20,2076,1)))
    model.add(tf.keras.layers.Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

# #create model
# model = tf.keras.Sequential()

# #add model layers
# model.add(tf.keras.layers.Conv2D(64, kernel_size=(2,2), activation='relu', input_shape=(20,2076,1)))
# model.add(tf.keras.layers.Conv2D(32, kernel_size=(2,2), activation='relu'))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
model=get_model()
#train model
optimizer = tf.keras.optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer=optimizer,metrics=['accuracy'])
model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=3, batch_size=2)
preds=model.predict(X_test)
print(preds)
print("=")