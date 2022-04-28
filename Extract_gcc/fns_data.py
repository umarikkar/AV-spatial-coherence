import os, glob
import numpy as np
import soundfile as sf
import simpleaudio as sa


def read_audio_file(sequence, rig, base_path, play_audio=False, normalise=False):
    
    # ==================== read audio file ===============================
    sequence_path = os.path.join(base_path, sequence, rig,'')
    
    # sequence_path = base_path + sequence + '/' + rig + '/'

    if rig == '01':
        chan_idx = np.array(
            ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16'])
    else:  # rig == '02'
        chan_idx = np.array(
            ['23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38'])

    audio = []
    for i in range(len(chan_idx)):
        # dirs = os.listdir(sequence_path)
        # a = [name for name in os.listdir(sequence_path) if name.endswith(".txt")][0]

        seq = sorted(glob.glob(sequence_path + chan_idx[i] + '-*.wav'))[0]
        aud, sampling_rate = sf.read(seq)

        if play_audio:
            wave_obj = sa.WaveObject.from_wave_file(seq)
            play_obj = wave_obj.play()
            play_obj.wait_done()

        audio.append(aud)

    audio = np.transpose(np.array(audio))  # (samples, channels)

    if normalise:
        audio = (audio - np.mean(audio, axis=0)) / np.std(audio, axis=0)

    return audio, sampling_rate