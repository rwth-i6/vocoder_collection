import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
import soundfile as sf
import sys
sys.path.append("/u/schuemann/experiments/tts_asr_2021/recipe/returnn_new")
from returnn.datasets.util.feature_extraction import _get_audio_db_mel_filterbank, _get_audio_features_mfcc, \
    _get_audio_log_mel_filterbank, _get_audio_log_log_mel_filterbank, _get_audio_linear_spectrogram
MAX_WAV_VALUE = 32768.0


def load_audio(full_path):
    data, sampling_rate = sf.read(full_path)
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


def extract_features(feature_name, audio, sr, win_size, hop_size, num_ff, f_min, f_max, num_mels, center, min_amp):
    # pad audio to correct length
    pad_hop = (sr * hop_size)
    pad_ff = (sr * win_size)
    audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0)
    audio = torch.nn.functional.pad(audio, (int((pad_ff - pad_hop) / 2), int((pad_ff - pad_hop) / 2)), mode='reflect')
    audio = audio.squeeze(0).squeeze(0)
    audio = np.array(audio)

    if feature_name == "mfcc":
        feature_data = _get_audio_features_mfcc(audio, sr, win_size, hop_size, num_ff, num_mels, f_min, f_max, center)
    elif feature_name == "log_mel_filterbank":
        feature_data = _get_audio_log_mel_filterbank(audio, sr, win_size, hop_size, num_ff)
    elif feature_name == "log_log_mel_filterbank":
        feature_data = _get_audio_log_log_mel_filterbank(audio, sr, win_size, hop_size, num_ff)
    elif feature_name == "db_mel_filterbank":
        feature_data = _get_audio_db_mel_filterbank(audio, sr, win_size, hop_size, num_ff, f_min, f_max, min_amp,
                                                    center)
    elif feature_name == "linear_spectrogram":
        feature_data = _get_audio_linear_spectrogram(audio, sr, win_size, hop_size, num_ff, center)
    return feature_data


def get_dataset_filelist(args):
    with open(args.train_file, 'r', encoding='utf-8') as fi:
        training_files = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    with open(args.valid_file, 'r', encoding='utf-8') as fi:
        validation_files = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, audio_format, training_files, segment_size, num_ff,
                 hop_size, win_size, sampling_rate, input_wav, output_wav,  fmin=60, fmax=7600, split=True, shuffle=True,
                 n_cache_reuse=1, device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None,
                 features="db_mel_filterbank", center=False, num_mels=128, min_amp=1e-10):
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
        self.features = features
        self.input_wavs_dir = input_wav
        self.output_wavs_dir = output_wav
        self.min_amp = min_amp
        self.audio_format = audio_format

    def __getitem__(self, index):
        filename = self.audio_files[index]
        input_path = os.path.join(self.input_wavs_dir, filename + self.audio_format)
        output_path = os.path.join(self.output_wavs_dir, filename + self.audio_format)
        if self._cache_ref_count == 0:
            input_audio, sampling_rate = load_audio(input_path)
            input_audio = input_audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                input_audio = normalize(input_audio) * 0.95
            self.cached_input_wav = input_audio

            output_audio, sampling_rate_ = load_audio(output_path)
            output_audio = output_audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                output_audio = normalize(output_audio) * 0.95
            self.cached_output_wav = output_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            input_audio = self.cached_input_wav
            output_audio = self.cached_output_wav
            self._cache_ref_count -= 1

        input_audio = torch.FloatTensor(input_audio)
        input_audio = input_audio.unsqueeze(0)

        output_audio = torch.FloatTensor(output_audio)
        output_audio = output_audio.unsqueeze(0)

        assert input_audio.size(1) == output_audio.size(1), "Inconsistent dataset length, unable to sampling"

        if self.split:
            if input_audio.size(1) >= self.segment_size:
                max_audio_start = input_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                input_audio = input_audio[:, audio_start:audio_start+self.segment_size]
                output_audio = output_audio[:, audio_start:audio_start+self.segment_size]
            else:
                input_audio = torch.nn.functional.pad(input_audio, (0, self.segment_size - input_audio.size(1)),
                                                      'constant')
                output_audio = torch.nn.functional.pad(output_audio, (0, self.segment_size - output_audio.size(1)),
                                                       'constant')

        mel = extract_features(self.features, np.array(output_audio.squeeze()), self.sampling_rate, self.win_size,
                               self.hop_size, self.num_ff, self.fmin, self.fmax, self.num_mels, self.center,
                               self.min_amp)
        mel = np.swapaxes(mel, 0, 1)
        mel = np.expand_dims(mel, axis=0)

        mel_loss = extract_features(self.features, np.array(output_audio.squeeze()), self.sampling_rate, self.win_size,
                                    self.hop_size, self.num_ff, self.fmin, self.fmax_loss, self.num_mels, self.center,
                                    self.min_amp)
        mel_loss = np.swapaxes(mel_loss, 0, 1)
        mel_loss = np.expand_dims(mel_loss, axis=0)

        return (input_audio.squeeze(0), output_audio.squeeze(0), filename, mel.squeeze(), mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
