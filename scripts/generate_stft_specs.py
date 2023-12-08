import argparse
import librosa
import numpy as np
from tqdm import tqdm
import os
import random
import sys
sys.path.append('/home/yuriih/AudioLDM')
import audioldm
from audioldm.audio import wav_to_fbank, TacotronSTFT, read_wav_file
from audioldm.audio.tools import get_mel_from_wav
import torch
from scipy.signal import butter, filtfilt

DEBUG = False

def add_delay(audio, sr, feedback_time, gain=0.5):
    delay_samples = int(sr * feedback_time)  

    # Create a delayed version of the signal
    delayed_signal = np.zeros_like(audio)
    delayed_signal[delay_samples:] = audio[:-delay_samples] * gain
    # Mix the original signal with the delayed signal
    y_with_delay = audio + delayed_signal
    return y_with_delay

def clip(x, level):
    return np.clip(x, -level, level)

def add_distortion(audio, sr, distortion_level=0.05):
    return clip(audio, distortion_level)

def low_pass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y.copy()

def high_pass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y.copy()

def _pad_spec(fbank, target_length=1024):
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    if fbank.size(-1) % 2 != 0:
        fbank = fbank[..., :-1]

    return fbank

def get_hifi_mel(waveform, target_length, fn_STFT):
    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    fbank = torch.FloatTensor(fbank.T)
    fbank = _pad_spec(fbank, target_length)
    return fbank
    

def main():

    if not DEBUG:
        parser = argparse.ArgumentParser()
        parser.add_argument('dataset_type', type=str, help='Dataset type (valid or test)')
        parser.add_argument('transform_type', type=str, help='Transform type (stft, cqt, mel, or hifi)', choices=['stft', 'cqt', 'mel', 'hifi'])
        args = parser.parse_args()
    else:
        class Args:
            def __init__(self, transform_type, dataset_type):
                self.transform_type = transform_type
                self.dataset_type = dataset_type

        args = Args('hifi', 'valid')

    audio_dir = ('LLTM/' if DEBUG else '') + f'datasets/nsynth-subset/nsynth-{args.dataset_type}-audios/audio'
    instrument_type = 'guitar'
    sr = 16000
    n_fft = 1024  # Number of Fourier components
    hop_length = 512  # Number of samples between successive frames
    window = 'hann' 

    stft_list = []
    file_names = []
    params_list = []
    i = 0
    # Search for an audio file that contains the instrument type in its name
    for filename in tqdm(os.listdir(audio_dir)):
        if filename.endswith('.wav'):
            audio_path = os.path.join(audio_dir, filename)
            y, _ = librosa.load(audio_path, sr=sr)
            
            # Augment
            # feedback_time = random.uniform(0.01, 0.1)
            # delay_level = random.uniform(0.1, 0.9)
            # distortion_level = random.uniform(0.01, 0.1)
            # y_low_pass = add_delay(y, sr, feedback_time, gain=delay_level)
            # y_high_pass = add_distortion(y_low_pass, sr, distortion_level)
            cutoff_low = random.randint(400, 700)
            cutoff_high= random.randint(1000, 1300)
            params_list.append([cutoff_high, cutoff_low])
            y_low_pass = low_pass_filter(y, cutoff=cutoff_low, fs=sr, order=5)
            y_high_pass = high_pass_filter(y, cutoff=cutoff_high, fs=sr, order=5)

            if args.transform_type == 'stft':
                D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
                D_high_pass = librosa.stft(y_high_pass, n_fft=n_fft, hop_length=hop_length, window=window)
                D_low_pass = librosa.stft(y_low_pass, n_fft=n_fft, hop_length=hop_length, window=window)
            elif args.transform_type == 'cqt':
                hop_length = 128
                # The lowest frequency of the CQT spectogram
                fmin = 32.7
                fmax = sr // 2
                # Max frequency resolution for 16000 sr
                n_bins = 95
                bins_per_octave = int(np.ceil(n_bins / np.log2(fmax / fmin)))

                D = librosa.cqt(y, sr=sr, hop_length = hop_length, fmin=fmin, window=window, n_bins=n_bins)
                D_high_pass = librosa.cqt(y_high_pass, sr=sr, hop_length = hop_length, fmin=fmin, window=window, n_bins=n_bins)
                D_low_pass = librosa.cqt(y_low_pass, sr=sr, hop_length = hop_length, fmin=fmin, window=window, n_bins=n_bins)
            elif args.transform_type == 'mel':
                n_fft = 1024
                hop_length = 256 
                n_mels=256

                D = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
                D_high_pass = librosa.feature.melspectrogram(y=y_high_pass, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
                D_low_pass = librosa.feature.melspectrogram(y=y_low_pass, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            
            elif args.transform_type == 'hifi':
                fn_STFT = TacotronSTFT(
                    filter_length=1024,
                    hop_length=160,
                    win_length=1024,
                    n_mel_channels=64,
                    sampling_rate=16000,
                    mel_fmin=0,
                    mel_fmax=8000,
                )
                duration = 4
                target_length=int(duration * 102.4)
                D_low_pass = get_hifi_mel(y_low_pass, target_length, fn_STFT).cpu().numpy()
                D_high_pass = get_hifi_mel(y_high_pass, target_length, fn_STFT).cpu().numpy()
                D = get_hifi_mel(y, target_length, fn_STFT).cpu().numpy()

            


            # Append the results to the lists
            stft_list.extend([np.array([D, D_high_pass, D_low_pass])])
            #file_names.extend([['clean', 'highpass', 'lowpass']])
            i+=1

    stft_array = np.array(stft_list)
    params_array = np.array(params_list)
    #file_names_array = ['clean', 'highpass', 'lowpass']*stft_array.shape[0]#','.join(file_names)

    print(f'Saving {i} files for {args.dataset_type} dataset...')
    np.save(f'datasets/effects/nsynth-{args.dataset_type}-{args.transform_type}_guitar_with_params.npy', stft_array)
    np.save(f'datasets/effects/eq_params.npy', params_array)
    # with open(('LLTM/' if DEBUG else '') + f'datasets/effects/nsynth-{args.dataset_type}-types_guitar.txt', 'w') as f:
    #     f.write(file_names_array)
    
if __name__ == "__main__":
    main()