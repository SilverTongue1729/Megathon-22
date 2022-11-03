def mean(l):
    return sum(l)/len(l)
import json
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#for loading and visualizing audio files
import librosa
import librosa.display
import numpy as np
from numpy import asarray, save, load
#to play audio
import IPython.display as ipd
import tensorflow as tf

accent_fpath = "train_wav"
accent_clips = os.listdir(accent_fpath)
max_audiosig=3000
X_train=np.array([])
X_test=np.array([])
y_train=np.array([])
y_test=np.array([])

MFCC_NUM=13
accent=0
A_NUM=15   # increase this to test more data
d={}
store_acc_name=[]
for i in accent_clips[:A_NUM]:
    print(accent)
    d[accent]=i
    audio_fpath = "train_wav/"+i
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
        if(c<4):
            X_train=np.append(X_train,delta_mfcc)
            y_train=np.append(y_train,accent)
            
        else:
            store_acc_name.append(i)
            X_test=np.append(X_test,delta_mfcc)
            y_test=np.append(y_test,accent)
        c+=1
        c=c%5
    accent+=1

y_train=tf.keras.utils.to_categorical(y_train)
y_test=tf.keras.utils.to_categorical(y_test)
extra_test_len = A_NUM-y_test.shape[1]
y_test = np.pad(y_test, pad_width=((0, 0), (0, extra_test_len)), mode='constant')

X_train = X_train.reshape(X_train.shape[0]//(MFCC_NUM*max_audiosig),MFCC_NUM,max_audiosig,1)
X_test = X_test.reshape(X_test.shape[0]//(MFCC_NUM*max_audiosig),MFCC_NUM,max_audiosig,1)



# Save time while finding mfcc of large datasets by storing in a binary .npy file

# save('X_train.npy', X_train)
# save('X_test.npy', X_test)
# save('y_train.npy', y_train)
# save('y_test.npy', y_test)

# X_train=load('X_train.npy')
# X_test=load('X_test.npy')
# y_train=load('y_train.npy')
# y_test=load('y_test.npy')

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)


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
    model.add(tf.keras.layers.Dense(A_NUM, activation='softmax'))
    return model


model=get_model()
#train model
optimizer = tf.keras.optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer=optimizer,metrics=['accuracy'])
model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=10, batch_size=2)     # Modify epochs and batch_size (increase it)


# To test on our X_test
# preds=model.predict(X_test)
# corr=0
# wrong=0
# for i in range(y_test.shape[0]):
#     print(store_acc_name[i],d[preds[i].argmax()])
#     if(store_acc_name[i]==d[preds[i].argmax()]):
#         corr+=1
#     else:
#         wrong+=1



# To test on given X_test
input_accent_fpath = "test_wav"
input_accent_clips = os.listdir(input_accent_fpath)
for j in input_accent_clips:
    input_clip_fpath = input_accent_fpath+"/"+j
    y, sr = librosa.load(input_clip_fpath, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_NUM)
    if (max_audiosig > mfcc.shape[1]):
            pad_width = max_audiosig - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            delta_mfcc=librosa.feature.delta(mfcc)
    input_X_test=np.append(input_X_test,delta_mfcc)
input_X_test = input_X_test.reshape(input_X_test.shape[0]//(MFCC_NUM*max_audiosig),MFCC_NUM,max_audiosig,1)
preds=model.predict(input_X_test)
corr=0
wrong=0
ans={}
for i in range(input_X_test.shape[0]):
    ans["test_"+str(i+1)]=d[preds[i].argmax()]
    print("test_",i+1,d[preds[i].argmax()])
fw=open("ground_truth.json","w")
json.dump(ans,fw)
    
