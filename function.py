import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.mlab import find

def parabolic(f, x):
    xv = 1/2 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv) 

def attributes(sample_rate,  samples):
    sig =(samples-np.mean(samples))/np.std(samples)
    print(sig)
    
    plt.plot(sig)
    plt.show()
    
    attrib = []
    
    n=1024
    sig=abs(sig)
    l=len(sig)
    size=int(sample_rate*0.03)
    word=0
    pause=1
    threshold=0.4
    one=(np.ones(size))/size
    line=np.convolve(sig, one,  mode='same')
    
    plt.figure(1)
    plt.plot(line)
    plt.show()
    
    for i in range(0, l, n):
        if (line[i]>threshold and line[i-n]<threshold):
            word+=1
        elif (line[i]<threshold and line[i-n]>threshold):
            pause+=1
    col=word+pause
    print(col) 
    print('количество слов', word)
    print('количество пауз',pause) 
    fr_s=word/col
    print('частота слов',fr_s) 
    attrib.append(fr_s)
    fr_p=pause/col
    print('частота пауз',fr_p) 
    attrib.append(fr_p)
    
    time=[]
    for i in range(0,l):
        if (line[i]>threshold and line[i-1]<threshold):
            start=i
            j=i
            while line[j+1]>threshold:
                j+=1
                
            end=j
            time.append((end-start)/sample_rate)     
            
    sr_time=np.mean(time)    
    print('средняя длительность слов',sr_time)
    attrib.append(sr_time)
    
    ton = 0

    sig -= np.mean(sig)
    corr = signal.fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)//2:]
    print(corr)
    plt.plot(corr)
    plt.show()
    
    d = np.diff(corr)
    print(d)
    start = find(d > 0)[0]
    print(start)

    peak = np.argmax(corr[start:]) + start
    print(peak)
    px = parabolic(corr, peak)[0]
    print(px)
    ton = sample_rate / px
    print('частота основного тона', ton)
    attrib.append(ton)
    

    def get_wave_data(sample_rate, samples):
        assert sample_rate == SAMPLE_RATE, sample_rate
        if isinstance(samples[0], np.ndarray):
            samples = samples.mean(1)
        return samples

    def show_specgram(samples):
        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        ax.specgram(samples,
            NFFT=WINDOW_SIZE, noverlap=WINDOW_SIZE - WINDOW_STEP, Fs=SAMPLE_RATE)
        plt.show() 
    samples = get_wave_data(sample_rate, sig)
    show_specgram(sig) 
    return attrib

WINDOW_SIZE = 2048
WINDOW_STEP = 512
SAMPLE_RATE = 48000
sample_rate, samples = read('usr0024_male_adult_001.wav')
attributes(sample_rate,samples)        
    
   