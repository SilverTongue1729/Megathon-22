def mean(l):
    return sum(l)/len(l)

import os
import matplotlib.pyplot as plt

#for loading and visualizing audio files
import librosa
import librosa.display
import numpy as np
#to play audio
import IPython.display as ipd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
accent_fpath = "/home/roja_26/Year2/megathon/Megathon-Proj/train_wav"
accent_clips = os.listdir(accent_fpath)
max_audiosig=2076
X_train=[]
y_train=[]
lang=[0]*10
c=0
for i in accent_clips:
    audio_fpath = "/home/roja_26/Year2/megathon/Megathon-Proj/train_wav/"+i
    audio_clips = os.listdir(audio_fpath)
    lang[c]=1
    if(len(audio_clips)>=29):
    # print(i)
    # print("No. of .wav files in audio folder = ",len(audio_clips))
        y_train.append(lang)
        for j in audio_clips[0:2]:
            clip_fpath = audio_fpath+"/"+j
            # print(clip_fpath)
            # x, sr = librosa.load(clip_fpath, sr=22050)
            # #  Waveform
            # print(type(x), type(sr))
            # print(x.shape, sr)
            # plt.figure(figsize=(14, 5))
            # librosa.display.waveshow(x, sr=sr)
            # plt.show()

            
            # X = librosa.stft(x)
            # Xdb = librosa.amplitude_to_db(abs(X))
            
            # # Normal Spectogram
            # plt.figure(figsize=(14, 5))
            # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
            # plt.colorbar()
            # plt.show()

            # # Log spectogram
            # plt.figure(figsize=(14, 5))
            # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
            # plt.colorbar()
            # plt.show()

            # Mel spectogram
            # plt.figure(figsize=(14, 5))
            # mel_spect = librosa.feature.melspectrogram(Xdb, sr=sr, n_fft=2048, hop_length=1024)
            # mel_spect = librosa.power_to_db(Xdb, ref=np.max)
            # librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time')
            # plt.colorbar()
            # plt.show()

            # MFCC
            # plt.figure(figsize=(14, 5))
            y, sr = librosa.load(clip_fpath, sr=22050)
            # print(type(y), type(y))
            # print(y.shape, sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
            X_train.append(mfcc)
            # if mfcc.shape[1]>max_audiosig:
            #    max_audiosig =  mfcc.shape[1]

            # librosa.display.specshow(mfcc, x_axis="time")
            # print(mfcc.shape)
            # plt.colorbar()
            # plt.show()
        c+=1




