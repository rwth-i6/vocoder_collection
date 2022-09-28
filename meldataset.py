import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import sys
sys.path.append("/u/schuemann/experiments/tts_asr_2021/recipe/returnn_new")
from returnn.datasets.util.feature_extraction import _get_audio_db_mel_filterbank, _get_audio_features_mfcc, \
    _get_audio_log_mel_filterbank, _get_audio_log_log_mel_filterbank, _get_audio_linear_spectrogram
MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def extract_features(feature_name, audio, sr, win_size, hop_size, num_ff, f_min, f_max, num_mels, center):
    if feature_name == "mfcc":
        feature_data = _get_audio_features_mfcc(audio, sr, win_size, hop_size, num_ff, num_mels, f_min, f_max, center)
    elif feature_name == "log_mel_filterbank":
        feature_data = _get_audio_log_mel_filterbank(audio, sr, win_size, hop_size, num_ff)
    elif feature_name == "log_log_mel_filterbank":
        feature_data = _get_audio_log_log_mel_filterbank(audio, sr, win_size, hop_size, num_ff)
    elif feature_name == "db_mel_filterbank":
        feature_data = _get_audio_db_mel_filterbank(audio, sr, win_size, hop_size, num_ff, f_min, f_max, center)
    elif feature_name  == "linear_spectrogram":
        feature_data = _get_audio_linear_spectrogram(audio, sr, win_size, hop_size, num_ff, center)
    return feature_data


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, num_ff,
                 hop_size, win_size, sampling_rate,  fmin=60, fmax=7600, split=True, shuffle=True,
                 n_cache_reuse=1, device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None,
                 features="db_mel_filterbank", center=False, num_mels=128):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.num_ff = num_ff
        self.center = center
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.largest_seq = 808
        self.features = features
    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            #if sampling_rate != self.sampling_rate:
            #    raise ValueError("{} SR doesn't match target {} SR".format(
            #        sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
            
            # compute mels for training
            mel = extract_features(self.features, np.array(audio.squeeze()), self.sampling_rate, self.win_size,
                                   self.hop_size, self.num_ff, self.fmin, self.fmax, self.num_mels, self.center)
            mel = np.swapaxes(mel, 0, 1)
            mel = np.expand_dims(mel, axis=0)
            
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        # compute mels for loss computation
        mel_loss = extract_features(self.features, np.array(audio.squeeze()), self.sampling_rate, self.win_size,
                                    self.hop_size, self.num_ff, self.fmin, self.fmax_loss, self.num_mels, self.center)
        mel_loss = np.swapaxes(mel_loss, 0, 1)
        mel_loss = np.expand_dims(mel_loss, axis=0)

        # compute noise based on mel shape for generation of fake audio
        noise = torch.randn([64, mel.shape[-1]])
        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze(), noise)

    def __len__(self):
        return len(self.audio_files)
