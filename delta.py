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

accent_fpath = "/home/roja_26/Year2/megathon/Megathon-Proj/train_wav"
accent_clips = os.listdir(accent_fpath)
max_audiosig=3000
X_train=np.array([])
X_test=np.array([])
y_train=np.array([])
y_test=np.array([])

# y_train = y_train.reshape(20)
# y_train = tf.convert_to_tensor(y_train)



MFCC_NUM=13
accent=0
for i in accent_clips:
    audio_fpath = "/home/roja_26/Year2/megathon/Megathon-Proj/train_wav/"+i
    audio_clips = os.listdir(audio_fpath)
    c=0
    for j in audio_clips:
        clip_fpath = audio_fpath+"/"+j
        y, sr = librosa.load(clip_fpath, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_NUM)
        if (max_audiosig > mfcc.shape[1]):
            pad_width = max_audiosig - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            delta_mfcc=librosa.feature.delta(mfcc)
            print(delta_mfcc.shape)
        if(c<4):
            X_train=np.append(X_train,delta_mfcc)
            y_train=np.append(y_train,accent)
        else:
            X_test=np.append(X_test,delta_mfcc)
            y_test=np.append(y_test,accent)
        c+=1
        c=c%5
    accent+=1

y_train=tf.keras.utils.to_categorical(y_train)
y_test=tf.keras.utils.to_categorical(y_test)

X_train = X_train.reshape(X_train.shape[0]//(MFCC_NUM*max_audiosig),MFCC_NUM,max_audiosig,1)
X_test = X_test.reshape(X_test.shape[0]//(MFCC_NUM*max_audiosig),MFCC_NUM,max_audiosig,1)
# y_train = y_train.reshape(20)
# y_train = tf.keras.utils.to_categorical(y_train)


def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(MFCC_NUM,max_audiosig,1)))
    model.add(tf.keras.layers.Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(122, activation='softmax'))
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
preds=model.predict(X_test[:5])
print(preds)
print("=")