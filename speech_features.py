import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.mlab import find


def filtering(samples, sample_rate):
    # функция выполняет фильтрацию сигнала
    # принимает зашумлённые сэмплы и частоту дискретизации
    # возвращает то же количество сэмплов, но без уже шума
    # в простейшем случае возвращаем исходный сигнал
    return samples


def preprocess(samples, sample_rate):
    # предобработка сигнала
    # здесь может быть удаление пауз, участков тишины, не речи и т.д.
    # в простейшем случае возвращаем исходный сигнал
    return samples

def parabolic(f, x):

    xv = 1/2 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv) 

def features(samples, sample_rate):
    # фильтрация
    samples = filtering(samples, sample_rate)

    # предобработка
    samples = preprocess(samples, sample_rate)

    # расчёт признаков
    # здесь должен быть ваш собственный код по расчёту признаков для сигнала samples
    # функция должна возвращать признаки в виде вектора float-чисел
    # feats = ...  # <- тут ваш код
    sig =(samples-np.mean(samples))/np.std(samples)
    
    feats = []
    
    n=256
    sig=abs(sig)
    l=len(sig)
    size=int(sample_rate*0.03)
    word=0
    pause=1
    threshold=0.5
    one=(np.ones(size))/size
    line=np.convolve(sig, one,  mode='same')
    
    for i in range(0, l, n):
        if (line[i]>threshold and line[i-n]<threshold):
            word+=1
        elif (line[i]<threshold and line[i-n]>threshold):
            pause+=1
    col=word+pause
    #частота слов
    fr_s=word/col
    feats.append(fr_s)
    #частота пауз
    fr_p=pause/col
    feats.append(fr_p)
    
    time=[]
    for i in range(0,l):
        if (line[i]>threshold and line[i-1]<threshold):
            start=i
            j=i
            while line[j+1]>threshold:
                j+=1
                
            end=j
            time.append((end-start)/sample_rate)     
    #средняя длительность слов        
    sr_time=np.mean(time)
    feats.append(sr_time)
    #частота основного тона
    ton = 0
    corr = signal.fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)//2:]

    d = np.diff(corr)
    start = find(d > 0)[0]

    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    ton = sample_rate / px
    feats.append(ton)
    return feats
