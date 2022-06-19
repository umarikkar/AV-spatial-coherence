#!/usr/bin/python
import csv
import os
import random
from contextlib import suppress
from pathlib import Path

# import core.config as conf
import h5py
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.signal import convolve
from sklearn import preprocessing

import core.config as conf


def csv_to_list(csv_path):
    as_list = None
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        as_list = list(reader)
    return as_list


def generate_mel_spectrograms(audio, sr, winlen, hoplen, numcep, n_fft, fmin, fmax):
    # function that computes the mfcc

    # INPUTS:
    # audio: audio temporal signal
    # sr: sampling rate
    # winlen: window lenght for stft
    # hoplen: number of steps between adjacent stft
    # numcep: number of bins for mel filtering
    # n_fft: fft window length
    # fmin: cut off low frequencies
    # fmax: cut off high frequency

    # OUTPUTS:
    # logmel_spectrogram: log melspectrograms

    # tensor = np.expand_dims(mfcc, axis=0)

    melW = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=numcep,
        fmin=fmin,
        fmax=fmax).T

    ## COMPUTE STFT
    stft_matrix = librosa.core.stft(
        y=audio,
        n_fft=n_fft,
        hop_length=hoplen,
        win_length=winlen,
        window=np.hanning(winlen),
        center=True,
        dtype=np.complex64,
        pad_mode='reflect').T

    ## COMPUTE MEL SPECTROGRAM
    mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, melW)

    ## COMPUTE LOG MEL SPECTROGRAM
    logmel_spectrogram = librosa.core.power_to_db(mel_spectrogram, ref=1.0, amin=1e-10, top_db=None)
    logmel_spectrogram = logmel_spectrogram.astype(np.float32)

    ## INTERPOLATE SPECTROGRAM TO MATCH DESIRED TEMPORAL LENGTH (e.g. 960 intead of 961)
    logmel_spectrogram = interp_tensor(logmel_spectrogram, logmel_spectrogram.shape[0],
                                       np.round(conf.training_param['frame_len_samples'] / conf.logmelspectro[
                                           'hoplen']))

    # ------------------- to show melspectrogram:
    '''
    fig, ax = plt.subplots()
    mfcc_data = np.swapaxes(logmel_spectrogram, 0, 1)
    #cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')#, vmin=-120,
    #                #vmax=120)
    #ax.set_title('MFCC')
    #fig.colorbar(cax)
    #plt.show()

    img = librosa.display.specshow(mfcc_data, sr=sr, y_axis='mel',hop_length=hoplen,x_axis='time', ax=ax)
    ax.set_title('MFCC')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()
    '''
    logmel_spectrogram = logmel_spectrogram[None, :, :]
    return logmel_spectrogram


def generate_gcc_spectrograms(audio, refaudio, winlen, hoplen, numcep, n_fft):
    # function that computes the mfcc

    # INPUTS:
    # audio: audio temporal signal
    # refaudio: reference channel
    # sr: sampling rate
    # winlen: window lenght for stft
    # hoplen: number of steps between adjacent stft
    # numcep: number of bins for mel filtering
    # n_fft: fft window length

    # OUTPUTS:
    # gcc_phat: log mel gcc-phat

    #ncorr = 2 * n_fft - 1
    #n_fft = int(2 ** np.ceil(np.log2(np.abs(ncorr))))
    Px = librosa.stft(y=audio,
                      n_fft=n_fft,
                      hop_length=hoplen,
                      center=True,
                      window=np.hanning(winlen),
                      pad_mode='reflect')
    Px_ref = librosa.stft(y=refaudio,
                          n_fft=n_fft,
                          hop_length=hoplen,
                          center=True,
                          window=np.hanning(winlen),
                          pad_mode='reflect')

    R = Px * np.conj(Px_ref)

    n_frames = R.shape[1]
    gcc_phat = []
    for i in range(n_frames):
        spec = R[:, i].flatten()
        cc = np.fft.irfft(np.exp(1.j * np.angle(spec)))
        cc = np.concatenate((cc[-numcep // 2:], cc[:numcep // 2]))
        gcc_phat.append(cc)
    gcc_phat = np.array(gcc_phat)
    #gcc_phat = gcc_phat.astype(np.float32)

    ## INTERPOLATE SPECTROGRAM TO MATCH DESIRED TEMPORAL LENGTH (e.g. 960 intead of 961)
    gcc_phat = interp_tensor(gcc_phat, gcc_phat.shape[0],
                      np.round(conf.training_param['frame_len_samples'] / conf.logmelspectro['hoplen']))

    # ------------------- to show gcc spectrogram:
    '''
    fig, ax = plt.subplots()
    gcc_data = np.swapaxes(gcc_phat, 0, 1)
    cax = ax.imshow(gcc_data, interpolation='nearest', cmap=cm.jet, origin='lower', extent=[0,960,-32,32],aspect='auto')#, vmin=-120,
                    #vmax=120)
    ax.set_title('GCC-PHAT')
    fig.colorbar(cax)
    plt.show()

    #img = librosa.display.specshow(gcc_data, sr=48000, y_axis='mel',hop_length=hoplen,x_axis='time', ax=ax)
    #ax.set_title('GCC-PHAT')
    #fig.colorbar(img, ax=ax, format="%+2.f dB")
    #plt.show()
    '''

    gcc_phat = gcc_phat[None, :, :]
    return gcc_phat


def calib_channels(x, array=None):
    # function that multiply the array channels by a gain factor in order to level their Root Mean Squared

    # INPUTS:
    # x: microphone array signals
    # array: AVA rig 1 or 2

    if array == 1:
        gains = conf.beamformer['gains_rig1']
        for i in range(len(gains)):
            x[:,i] = np.multiply(gains[i], x[:,i])
    elif array == 2:
        gains = conf.beamformer['gains_rig2']
        for i in range(len(gains)):
            x[:,i] = np.multiply(gains[i], x[:,i])
    else:
        print('CALIBRATION: neither rig 1 nor 2 are selected')

    return x


def BF_filt(x,w):
    # function that filters the input signal with the microphone array weights

    # INPUTS:
    # x: microphone array signals
    # w: beamforming weights

    # OUTPUTS:
    # y: beamformer output signal

    x_i = np.array(x[:, 0])
    w_i = np.array(w[:, 0])
    y = convolve(x_i, w_i, mode='full', method='fft')

    for i in range(1,16): # 16 number of mics
        x_i = np.array(x[:,i])
        w_i = np.array(w[:,i])

        y_i = convolve(x_i, w_i, mode='full', method='fft')
        y = np.vstack((y,y_i))

    y = np.sum(np.transpose(y), axis=1)
    #y = 0.999 * y / np.max(abs(y)) # normalise
    y = y[:-(len(w_i)-1)]

    return y


def pad_audio_clip(audio, desired_length):
    if len(audio) < desired_length: # signal too short
        #print(len(audio))
        return np.concatenate((audio,np.zeros(desired_length - len(audio))))
    else: # signal correct or too long
        return audio[0: desired_length]


def norm(x, mean, std):
    return (x - mean) / std

def cam_one_hot(cam):
    one_hot = np.zeros(11) # 11 cameras
    one_hot[cam-1] = 1
    return one_hot # torch.from_numpy(one_hot)


def get_latest_ckpt(path, reverse=False, suffix='.ckpt'):
    """Load latest checkpoint from target directory. Return None if no checkpoints are found."""
    path, file = Path(path), None
    files = (f for f in sorted(path.iterdir(), reverse=not reverse) if f.suffix == suffix)
    with suppress(StopIteration):
        file = next(f for f in files)
    return file


def find(array, value): #returns a list with the indices of the positions of array = value
    array = np.asarray(array)
    a = np.where(array==value)
    x = []
    for item in a:
        x.extend(item)
    return x


def find_audio_frame_idx(csv_list, step): # returns a list with the indices of the first frame of the audio segments
    array = np.asarray(csv_list)[:, 1] # extract time array
    array = array.astype(np.float)
    #a = np.where((np.mod(array, step) == 0) & (array != 0))
    a = np.where((np.mod(array, step) == 0))
    x = []
    for item in a:
        x.extend(item)
    return x


def interp_tensor(tensor, tensor_frames_num, desired_frames_num=960, d_type=float):
    '''
    Interpolate tensor

    Args:
        tensor: (time_steps, frequency_bins)
        tensor_frames_num: tensor's temporal dimension
        desired_frames_num: desired tensor's temporal dimension
    '''
    ratio = 1.0 * desired_frames_num / tensor_frames_num

    new_len = int(np.around(ratio * tensor.shape[0]))
    new_tensor = np.zeros((new_len, tensor.shape[1]), dtype=d_type)

    for n in range(new_len):
        new_tensor[n] = tensor[int(np.around(n / ratio))]

    return new_tensor

def n_fold_generator(dataset_list, fold_num=5):
    size = len(dataset_list)
    random.shuffle(dataset_list)
    folded_dataset = list()
    fold_size = size // fold_num
    for i in range(0, size, fold_size):
        folded_dataset.append(dataset_list[i:i + fold_size])

    return folded_dataset

def belong_to_val(x, fold):
    condition_true = False
    for i in range(len(fold)):
        if x == fold[i]:
            condition_true = True
    return condition_true


def compute_scaler(h5_file, h5py_dir, is_salsa=False) -> None:
    """
    Compute feature mean and std vectors of spectrograms for normalization.
    :param h5file_dir: Feature directory that contains train and test folder.
    """
    print('============> Start calculating scaler')
    # Get the dimensions of feature by reading one feature files
    afeature = h5_file['features'][0]  # (n_channels, n_timesteps, n_features)

    n_channels = afeature.shape[0]
    if is_salsa: # only the spectrograms need to be normalized
        n_feature_channels = 1  # hard coded number (correspond to number of spectrogram channels)
    else:
        n_feature_channels = n_channels

    # initialize scaler
    scaler_dict = {}
    for i_chan in np.arange(n_feature_channels):
        scaler_dict[i_chan] = preprocessing.StandardScaler()

    # Iterate through data
    for idx in range(h5_file['features'].shape[0] // 1):
        afeature = h5_file['features'][idx]  # (n_channels, n_timesteps, n_features)
        for i_chan in range(n_feature_channels):
            scaler_dict[i_chan].partial_fit(afeature[i_chan, :, :])  # (n_timesteps, n_features)

    # Extract mean and std
    feature_mean = []
    feature_std = []
    for i_chan in range(n_feature_channels):
        feature_mean.append(scaler_dict[i_chan].mean_)
        feature_std.append(np.sqrt(scaler_dict[i_chan].var_))

    feature_mean = np.array(feature_mean)
    feature_std = np.array(feature_std)
    # Expand dims for timesteps: (n_chanels, n_timesteps, n_features)
    feature_mean = np.expand_dims(feature_mean, axis=1)
    feature_std = np.expand_dims(feature_std, axis=1)
    # Save scaler file
    scaler_path = os.path.join(str(h5py_dir) + '/feature_scaler.h5')
    print(scaler_path)
    with h5py.File(scaler_path, 'w') as hf:
        hf.create_dataset('mean', data=feature_mean, dtype=np.float32)
        hf.create_dataset('std', data=feature_std, dtype=np.float32)

    print('Scaler path: {}'.format(scaler_path))

def load_feature_scaler(h5_path):
    """
    Load feature scaler for multichannel spectrograms
    """
    with h5py.File(h5_path, 'r') as hf:
        mean = hf['mean'][:]
        std = hf['std'][:]
    return mean, std

