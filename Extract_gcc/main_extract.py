#!/usr/bin/python
import os
import glob
import soundfile as sf
import numpy as np
from gcc_phat_extraction import generate_gcc_spectrograms, generate_mel_spectrograms

# modify as appropriate
# base_path = '/home/davide/PycharmProjects/Extract_gcc/'
base_path = '/Users/umar_m/Projects/MSc-project/AV-spatial-coherence/Extract_gcc'

rig = '01'  # chose between rig '01' and rig '02'
seq = 'interactive1_t3'

# config metaparameters
winlen = 512  # samples
hoplen = 125  # samples
numcep = 64  # number of cepstrum bins (frequency axis)
n_fft = 512  # samples
fmin = 40  # min freq cut off
fmax = 24000  # max freq cut off (no greater than sr/2)


def read_audio_file(sequence, rig):
    # ==================== read audio file ===============================
    sequence_path = base_path + sequence + '/' + rig + '/'

    if rig == '01':
        chan_idx = np.array(
            ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16'])
    else:  # rig == '02'
        chan_idx = np.array(
            ['23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38'])

    audio = []
    for i in range(len(chan_idx)):
        seq = sorted(glob.glob(sequence_path + chan_idx[i] + '-*.wav'))[i]
        aud, sr = sf.read(seq)
        audio.append(aud)

    audio = np.transpose(np.array(audio))  # (samples, channels)
    return audio, sr


def main():
    audio, sr = read_audio_file(seq, rig)

    ## ----------- GCC SPECTROGRAMS TENSOR -------------
    output_tensor = []  # gcc_tensor
    channel_num = audio.shape[1]

    # CHOOSE BETWEEN OPTION 1) OR 2)

    # 1) use all possible microphone pairs as in original paper
    # careful: num of combinations = n_mics*(n_mics-1)/2 so, for 16 mics -> 120 GCC channels!
    '''
    for n in range(channel_num):
        for m in range(n + 1, channel_num):
            output_tensor.append(generate_gcc_spectrograms(audio[:,m], audio[:,n], winlen, hoplen, numcep, n_fft))
    '''

    # 2) use reference mic (num of combinations = n_mics-1)
    ref_mic_id = 5  # hard coded param (5 represents the sixth microphone of the array, the central one)
    for n in range(channel_num):
        if not n == ref_mic_id:
            output_tensor.append(
                generate_gcc_spectrograms(audio[:, n], audio[:, ref_mic_id], winlen, hoplen, numcep, n_fft))

    ## ---------- ADD mono log mel spect (1st channel only) ------------------------
    # note: the original paper computes the logmel spec for each channel of the 4-element tetrahedral array
    # for our array, one channel should be enough
    logmel = generate_mel_spectrograms(audio[:, 0], sr, winlen, hoplen, numcep, n_fft, fmin, fmax)
    output_tensor.append(logmel)

    output_tensor = np.concatenate(output_tensor, axis=0)  # (n_channels, timebins, freqbins)


if __name__ == "__main__":
    main()
