import glob
import librosa
import numpy as np
from sklearn.cluster import KMeans

p_data_wav_train=[]
t_data_wav_train=[]
k_data_wav_train=[]
b_data_wav_train=[]
d_data_wav_train=[]
g_data_wav_train=[]

for i in range(1,4):
    p_data_path_wav_train = 'PTKBDG/*/*p*' + str(i) + '.wav'
    p_data_wav_train = p_data_wav_train + glob.glob(p_data_path_wav_train)
    t_data_path_wav_train = 'PTKBDG/*/*t*' + str(i) + '.wav'
    t_data_wav_train = t_data_wav_train + glob.glob(t_data_path_wav_train)
    k_data_path_wav_train = 'PTKBDG/*/*k*' + str(i) + '.wav'
    k_data_wav_train = k_data_wav_train + glob.glob(k_data_path_wav_train)
    b_data_path_wav_train = 'PTKBDG/*/*b*' + str(i) + '.wav'
    b_data_wav_train = b_data_wav_train + glob.glob(b_data_path_wav_train)
    d_data_path_wav_train = 'PTKBDG/*/*d*' + str(i) + '.wav'
    d_data_wav_train = d_data_wav_train + glob.glob(d_data_path_wav_train)
    g_data_path_wav_train = 'PTKBDG/*/*g*' + str(i) + '.wav'
    g_data_wav_train = g_data_wav_train + glob.glob(g_data_path_wav_train)
p_data_wav_train.sort()
t_data_wav_train.sort()
k_data_wav_train.sort()
b_data_wav_train.sort()
d_data_wav_train.sort()
g_data_wav_train.sort()
# print(p_data_wav_train)
print(len(p_data_wav_train))
print(len(t_data_wav_train))
print(len(k_data_wav_train))
print(len(b_data_wav_train))
print(len(d_data_wav_train))
print(len(g_data_wav_train))


p_data_wav_test=[]
t_data_wav_test=[]
k_data_wav_test=[]
b_data_wav_test=[]
d_data_wav_test=[]
g_data_wav_test=[]

for i in range(4,6):
    p_data_path_wav_test = 'PTKBDG/*/*p*' + str(i) + '.wav'
    p_data_wav_test = p_data_wav_test + glob.glob(p_data_path_wav_test)
    t_data_path_wav_test = 'PTKBDG/*/*t*' + str(i) + '.wav'
    t_data_wav_test = t_data_wav_test + glob.glob(t_data_path_wav_test)
    k_data_path_wav_test = 'PTKBDG/*/*k*' + str(i) + '.wav'
    k_data_wav_test = k_data_wav_test + glob.glob(k_data_path_wav_test)
    b_data_path_wav_test = 'PTKBDG/*/*b*' + str(i) + '.wav'
    b_data_wav_test = b_data_wav_test + glob.glob(b_data_path_wav_test)
    d_data_path_wav_test = 'PTKBDG/*/*d*' + str(i) + '.wav'
    d_data_wav_test = d_data_wav_test + glob.glob(d_data_path_wav_test)
    g_data_path_wav_test = 'PTKBDG/*/*g*' + str(i) + '.wav'
    g_data_wav_test = g_data_wav_test + glob.glob(g_data_path_wav_test)
p_data_wav_test.sort()
t_data_wav_test.sort()
k_data_wav_test.sort()
b_data_wav_test.sort()
d_data_wav_test.sort()
g_data_wav_test.sort()
# print(p_data_wav_test)
print(len(p_data_wav_test))
print(len(t_data_wav_test))
print(len(k_data_wav_test))
print(len(b_data_wav_test))
print(len(d_data_wav_test))
print(len(g_data_wav_test))


p_data_txt_train=[]
t_data_txt_train=[]
k_data_txt_train=[]
b_data_txt_train=[]
d_data_txt_train=[]
g_data_txt_train=[]

for i in range(1,4):
    p_data_path_txt_train = 'PTKBDG/*/*p*' + str(i) + '.txt'
    p_data_txt_train = p_data_txt_train + glob.glob(p_data_path_txt_train)
    t_data_path_txt_train = 'PTKBDG/*/*t*' + str(i) + '.txt'
    t_data_txt_train = t_data_txt_train + glob.glob(t_data_path_txt_train)
    k_data_path_txt_train = 'PTKBDG/*/*k*' + str(i) + '.txt'
    k_data_txt_train = k_data_txt_train + glob.glob(k_data_path_txt_train)
    b_data_path_txt_train = 'PTKBDG/*/*b*' + str(i) + '.txt'
    b_data_txt_train = b_data_txt_train + glob.glob(b_data_path_txt_train)
    d_data_path_txt_train = 'PTKBDG/*/*d*' + str(i) + '.txt'
    d_data_txt_train = d_data_txt_train + glob.glob(d_data_path_txt_train)
    g_data_path_txt_train = 'PTKBDG/*/*g*' + str(i) + '.txt'
    g_data_txt_train = g_data_txt_train + glob.glob(g_data_path_txt_train)
p_data_txt_train.sort()
t_data_txt_train.sort()
k_data_txt_train.sort()
b_data_txt_train.sort()
d_data_txt_train.sort()
g_data_txt_train.sort()
# print(p_data_txt_train)
print(len(p_data_txt_train))
print(len(t_data_txt_train))
print(len(k_data_txt_train))
print(len(b_data_txt_train))
print(len(d_data_txt_train))
print(len(g_data_txt_train))


p_data_txt_test=[]
t_data_txt_test=[]
k_data_txt_test=[]
b_data_txt_test=[]
d_data_txt_test=[]
g_data_txt_test=[]

for i in range(4,6):
    p_data_path_txt_test = 'PTKBDG/*/*p*' + str(i) + '.txt'
    p_data_txt_test = p_data_txt_test + glob.glob(p_data_path_txt_test)
    t_data_path_txt_test = 'PTKBDG/*/*t*' + str(i) + '.txt'
    t_data_txt_test = t_data_txt_test + glob.glob(t_data_path_txt_test)
    k_data_path_txt_test = 'PTKBDG/*/*k*' + str(i) + '.txt'
    k_data_txt_test = k_data_txt_test + glob.glob(k_data_path_txt_test)
    b_data_path_txt_test = 'PTKBDG/*/*b*' + str(i) + '.txt'
    b_data_txt_test = b_data_txt_test + glob.glob(b_data_path_txt_test)
    d_data_path_txt_test = 'PTKBDG/*/*d*' + str(i) + '.txt'
    d_data_txt_test = d_data_txt_test + glob.glob(d_data_path_txt_test)
    g_data_path_txt_test = 'PTKBDG/*/*g*' + str(i) + '.txt'
    g_data_txt_test = g_data_txt_test + glob.glob(g_data_path_txt_test)
p_data_txt_test.sort()
t_data_txt_test.sort()
k_data_txt_test.sort()
b_data_txt_test.sort()
d_data_txt_test.sort()
g_data_txt_test.sort()
# print(p_data_txt_test)
print(len(p_data_txt_test))
print(len(t_data_txt_test))
print(len(k_data_txt_test))
print(len(b_data_txt_test))
print(len(d_data_txt_test))
print(len(g_data_txt_test))


def string_to_matrix(string):
    line_split = list(string.split("\n"))
    matrix = []

    for item in line_split:
        line = []
        for data in item.split("\t"):
            line.append(data)
        matrix.append(line)

    return matrix

p=1
K=16
N=3

train_feature = []
train_class = []

for i in range(len(p_data_wav_train)):
    y, fs = librosa.load(p_data_wav_train[i], sr=None)
    p_txt_train =  open(p_data_txt_train[i], "r").read()
    time_stamp_list = string_to_matrix(p_txt_train)
    time_stamp = np.zeros(shape=(2, 2), dtype=float)
    time_stamp[0, 0] = float(time_stamp_list[0][0])
    time_stamp[0, 1] = float(time_stamp_list[0][1])
    time_stamp[1, 0] = float(time_stamp_list[1][0])
    time_stamp[1, 1] = float(time_stamp_list[1][1])
#     print(time_stamp)
    start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
    end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
    start_index = int(start_time*fs)
    end_index = int(end_time*fs)
    y = y[start_index:end_index]
#     print(len(y))
    hop = int(0.01*fs)
    win = int(0.02*fs)
    y_mfcc = (librosa.feature.mfcc(y=y, sr=fs, n_mfcc=39, win_length=win, hop_length=hop)).T
    for j in range(len(y_mfcc)):
        train_feature.append(y_mfcc[j,:])
#         print(y_mfcc.shape)
        train_class.append(1)

for i in range(len(t_data_wav_train)):
    y, fs = librosa.load(t_data_wav_train[i], sr=None)
    t_txt_train =  open(t_data_txt_train[i], "r").read()
    time_stamp_list = string_to_matrix(t_txt_train)
    time_stamp = np.zeros(shape=(2, 2), dtype=float)
    time_stamp[0, 0] = float(time_stamp_list[0][0])
    time_stamp[0, 1] = float(time_stamp_list[0][1])
    time_stamp[1, 0] = float(time_stamp_list[1][0])
    time_stamp[1, 1] = float(time_stamp_list[1][1])
#     print(time_stamp)
    start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
    end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
    start_index = int(start_time*fs)
    end_index = int(end_time*fs)
    y = y[start_index:end_index]
#     print(len(y))
    hop = int(0.01*fs)
    win = int(0.02*fs)
    y_mfcc = (librosa.feature.mfcc(y=y, sr=fs, n_mfcc=39, win_length=win, hop_length=hop)).T
    for j in range(len(y_mfcc)):
        train_feature.append(y_mfcc[j,:])
#         print(y_mfcc.shape)
        train_class.append(2)

for i in range(len(k_data_wav_train)):
    y, fs = librosa.load(k_data_wav_train[i], sr=None)
    k_txt_train =  open(k_data_txt_train[i], "r").read()
    time_stamp_list = string_to_matrix(k_txt_train)
    time_stamp = np.zeros(shape=(2, 2), dtype=float)
    time_stamp[0, 0] = float(time_stamp_list[0][0])
    time_stamp[0, 1] = float(time_stamp_list[0][1])
    time_stamp[1, 0] = float(time_stamp_list[1][0])
    time_stamp[1, 1] = float(time_stamp_list[1][1])
#     print(time_stamp)
    start_time = time_stamp[0, 1] - p*(time_stamp[0, 1] - time_stamp[0, 0])
    end_time = time_stamp[1, 0] + p * (time_stamp[1, 1] - time_stamp[1, 0])
    start_index = int(start_time*fs)
    end_index = int(end_time*fs)
    y = y[start_index:end_index]
#     print(len(y))
    hop = int(0.01*fs)
    win = int(0.02*fs)
    y_mfcc = (librosa.feature.mfcc(y=y, sr=fs, n_mfcc=39, win_length=win, hop_length=hop)).T
    for j in range(len(y_mfcc)):
        train_feature.append(y_mfcc[j,:])
#         print(y_mfcc.shape)
        train_class.append(3)

# print(np.array(train_feature).shape)
print(np.array(train_feature[1]).shape)
print(len(train_feature))
print(len(train_class))

kmeans = KMeans(init="random", n_clusters=K ,n_init=10, max_iter=300, random_state=42)
kmeans.fit(train_feature)
clusters = kmeans.cluster_centers_
print(clusters.shape)
print(clusters)





