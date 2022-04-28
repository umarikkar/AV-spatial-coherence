# !/usr/bin/python


import os
import glob
from tqdm import tqdm

import soundfile as sf
import simpleaudio as sa
import numpy as np
import matplotlib.pyplot as plt

from gcc_phat_extraction import generate_gcc_spectrograms, generate_mel_spectrograms
from fns_data import read_audio_file

# modify as appropriate
# base_path = '/home/davide/PycharmProjects/Extract_gcc/'
# base_path = '/Users/umar_m/Projects/MSc-project/AV-spatial-coherence/Extract_gcc/'
base_path = os.path.dirname(os.path.abspath(__file__))

rig = '01'  # chose between rig '01' and rig '02'
seq = 'interactive1_t3'
play_audio = False
plot_spectrogram = True

# config metaparameters
winlen = 512  # samples
hoplen = 125  # samples
numcep = 64  # number of cepstrum bins (frequency axis)
n_fft = 512  # samples
fmin = 40  # min freq cut off
fmax = 24000  # max freq cut off (no greater than sr/2)

out_gcc = True
plot_multi = False
plot_mel=False
plot_gcc=False
plot_waves=False

def main():

    audio1, sr = read_audio_file(seq, rig, base_path, normalise=True)
    audio2, sr = read_audio_file(seq, '02', base_path, normalise=True)

    ## --- generate a 10 second clip -----------------------

    t = np.array(range(audio1.shape[0])) / sr
    t_min = 5
    t_max = t_min + 5

    n_min, n_max = t_min*sr, t_max*sr

    if plot_waves:
        plt.figure(figsize=(6,3))
        plt.plot(t, audio1[:,0])
        plt.plot(t, audio2[:,0])
        plt.xlim(t_min, t_max)
        plt.ylim(-3.5, 3.5)
        plt.legend(['Channel index: 1', 'Channel index: 5'])
        plt.xlabel('time/s')
        plt.ylabel('Amplitude')
        plt.show()
    
    audio = np.copy(audio1[n_min:n_max, :])

    output_tensor = []
    output_mag = []  # gcc_tensor
    output_phase = []
    channel_num = audio.shape[1]

    if out_gcc:
        ## ----------- GCC SPECTROGRAMS TENSOR -------------

        # CHOOSE BETWEEN OPTION 1) OR 2)

        # 1) use all possible microphone pairs as in original paper
        # careful: num of combinations = n_mics*(n_mics-1)/2 so, for 16 mics -> 120 GCC channels!
        '''
        for n in range(channel_num):
            for m in range(n + 1, channel_num):
                output_tensor.append(generate_gcc_spectrograms(audio[:,m], audio[:,n], winlen, hoplen, numcep, n_fft))
        '''

        # 2) use reference mic (num of combinations = n_mics-1)
        ref_mic_id = 0  # hard coded param (5 represents the sixth microphone of the array, the central one)
        for n in tqdm(range(channel_num)):
            if not n == ref_mic_id:
                output_tensor.append(
                    generate_gcc_spectrograms(audio[:, n], audio[:, ref_mic_id], winlen, hoplen, numcep, n_fft, plot_gcc=plot_gcc))

        # print(len(output_tensor))
        ## ---------- ADD mono log mel spect (1st channel only) ------------------------
        # note: the original paper computes the logmel spec for each channel of the 4-element tetrahedral array
        # for our array, one channel should be enough
        logmel, _ = generate_mel_spectrograms(audio[:, 0], sr, winlen, hoplen, numcep, n_fft, fmin, fmax)
        output_tensor.append(logmel)



        if plot_multi:
            for idx, out in enumerate(output_tensor):
                # print(idx)
                fig=plt.figure(figsize=(2,2), linewidth=5, edgecolor="#04253a")
                plt.imshow(out.T, origin='lower', aspect='auto', vmin=0, vmax=1)
                plt.axis('off')
                plt.savefig(os.path.join(base_path, 'figs', 'GCC_'+str(idx)+'.png'), bbox_inches='tight', edgecolor=fig.get_edgecolor())

        output_tensor = np.concatenate(output_tensor, axis=0)  # (n_channels, timebins, freqbins)

        plt.figure()
        plt.subplot(131)
        plt.imshow(output_tensor[0,:,:].T,  origin='lower', aspect='auto', vmin=0, vmax=1)
        plt.ylabel('lag (in samples)'), plt.xlabel('sample'), plt.title('(a)')
        plt.subplot(132)
        plt.imshow(output_tensor[4,:,:].T,  origin='lower', aspect='auto', vmin=0, vmax=1)
        plt.xlabel('sample'), plt.title('(b)')
        plt.subplot(133)
        plt.imshow(output_tensor[8,:,:].T,  origin='lower', aspect='auto', vmin=0, vmax=1)
        plt.xlabel('sample'), plt.title('(c)')
        plt.show()

        print('output_tensor:', output_tensor.shape)

    else:
        ## ----------- LOG-MEL SPECTROGRAMS -------------
        for n in tqdm(range(channel_num)):
            # print('running for mic:', n+1)
            mag, ph = generate_mel_spectrograms(audio[:, n], sr, winlen, hoplen, numcep, n_fft, fmin, fmax, plot_mel=plot_mel)
            output_mag.append(mag)
            output_phase.append(ph)

        if plot_multi:
            for idx, out in enumerate(output_phase):
                # print(idx)
                fig=plt.figure(figsize=(2,2), linewidth=5, edgecolor="#04253a")
                plt.imshow(out.T, origin='lower', aspect='auto')
                plt.axis('off')
                plt.savefig(os.path.join(base_path, 'figs', 'mel_'+str(idx)+'_phase.png'), bbox_inches='tight', edgecolor=fig.get_edgecolor())

        output_tensor = np.concatenate(output_mag, axis=0)  # (n_channels, timebins, freqbins)
    np.save(os.path.join(base_path, 'gcc_tensor.npy'), 'output_tensor')



if __name__ == "__main__":
    main()
