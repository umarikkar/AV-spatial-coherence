#!/usr/bin/python

'''
Credits to:
https://github.com/yinkalario/Two-Stage-Polyphonic-Sound-Event-Detection-and-Localization
'''

import numpy as np
import librosa
import matplotlib.pyplot as plt


def generate_mel_spectrograms(audio, sr, winlen, hoplen, numcep, n_fft, fmin, fmax, plot_mel=False):
    '''
    # function that computes the mfcc

    # INPUTS:
    # audio: audio temporal signal (single-channel)
    # sr: sampling rate
    # winlen: window length for stft
    # hoplen: number of steps between adjacent stft
    # numcep: number of bins for mel filtering
    # n_fft: fft window length
    # fmin: cut off low frequencies
    # fmax: cut off high frequency

    # OUTPUTS:
    # logmel_spectrogram: log melspectrograms
    '''
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

    phase_spectrogram = np.dot(np.angle(stft_matrix), melW)

    ## COMPUTE MEL SPECTROGRAM
    mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, melW)

    ## COMPUTE LOG MEL SPECTROGRAM
    logmel_spectrogram = librosa.core.power_to_db(mel_spectrogram, ref=1.0, amin=1e-10, top_db=None)
    logmel_spectrogram = logmel_spectrogram.astype(np.float32)

    if plot_mel:
        plt.figure(figsize=(15,5))
        plt.subplot(131)
        plt.imshow(np.abs(stft_matrix.T), origin='lower', aspect='auto',vmin=0,vmax=1)
        plt.xlabel('samples')
        plt.ylabel('frequency bin')
        plt.subplot(132)
        plt.imshow(mel_spectrogram.T, origin='lower', aspect='auto',vmin=0,vmax=1)
        plt.xlabel('samples')
        plt.ylabel('frequency bin')
        plt.subplot(133)
        plt.imshow(logmel_spectrogram.T, origin='lower', aspect='auto',vmin=0,vmax=1)
        plt.xlabel('samples')
        plt.ylabel('frequency bin')
        plt.show()

        plt.figure(figsize=(5,5))
        # plt.subplot(131)
        plt.imshow(phase_spectrogram.T, origin='lower', aspect='auto')
        plt.xlabel('samples')
        plt.ylabel('frequency bin')

        # plt.subplot(132)
        # plt.imshow(mel_spectrogram.T, origin='lower', aspect='auto',vmin=0,vmax=1)
        # plt.xlabel('samples')
        # plt.ylabel('frequency bin')

        # plt.subplot(133)
        # plt.imshow(logmel_spectrogram.T, origin='lower', aspect='auto',vmin=0,vmax=1)
        # plt.xlabel('samples')
        # plt.ylabel('frequency bin')

        plt.show()

    
    

    logmel_spectrogram = logmel_spectrogram[None, :, :]
    phase_spectrogram = phase_spectrogram[None, :, :]
    return logmel_spectrogram, phase_spectrogram


def generate_gcc_spectrograms(audio, refaudio, winlen, hoplen, numcep, n_fft, plot_gcc=False):
    '''
    # function that computes the mfcc

    # INPUTS:
    # audio: audio temporal signal (single-channel)
    # refaudio: reference channel
    # sr: sampling rate
    # winlen: window length for stft
    # hoplen: number of steps between adjacent stft
    # numcep: number of bins for mel filtering
    # n_fft: fft window length

    # OUTPUTS:
    # gcc_phat: log mel gcc-phat
    '''

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

    R1 = np.concatenate( ( R[-numcep//2:,:],  R[:numcep//2,:] ) , axis=0 ).T

    n_frames = R.shape[1]
    gcc_phat = []

    for i in range(n_frames):
        spec = R[:, i].flatten()
        cc = np.fft.irfft(np.exp(1.j * np.angle(spec)))
        cc = np.concatenate((cc[-numcep // 2:], cc[:numcep // 2]))
        gcc_phat.append(cc)
    gcc_phat = np.array(gcc_phat)

    if plot_gcc:
        plt.figure(figsize=(15,5))
        plt.subplot(121)
        plt.imshow(np.real(R), origin='lower', aspect='auto',vmin=0,vmax=1)
        plt.xlabel('samples')
        plt.ylabel('frequency bin')
        plt.subplot(122)
        plt.imshow(gcc_phat.T, origin='lower', aspect='auto',vmin=0,vmax=1)
        plt.xlabel('samples')
        plt.ylabel('time lag')
        plt.show()


    gcc_phat = gcc_phat[None, :, :]
    return gcc_phat
