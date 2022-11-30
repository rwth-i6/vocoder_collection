import math
import os
import random
import torch
import torch.utils.data
import soundfile as sf
import numpy as np
from librosa.util import normalize
import multiprocessing
import sys
import logging
sys.path.append("/u/schuemann/experiments/tts_asr_2021/recipe/returnn_new")
from returnn.datasets.basic import init_dataset
from returnn.datasets.util.feature_extraction import _get_audio_db_mel_filterbank, _get_audio_features_mfcc, \
    _get_audio_log_mel_filterbank, _get_audio_log_log_mel_filterbank, _get_audio_linear_spectrogram

MAX_WAV_VALUE = 32768.0
# need two seperate locks for sequence loading and data acquisition, otherwise deadlocks occur
load_lock = multiprocessing.Lock()
data_lock = multiprocessing.Lock()

def load_seqs(dataset, idx):
    dataset.load_seqs(idx,idx+1)
  
def get_audio(idx, dataset):
    data = dataset.get_input_data(idx)
    data = np.swapaxes(data,0,1)
    tag = dataset.get_tag(idx)
    return data, tag

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

def get_dataset(path, training_file, shuffle):
    # we always want raw audio as features for loss computation
    audio_opts = {"features":"raw", "peak_normalization":False}
    
    # targets are not needed in the dataset, just used for caching audio
    d = {
            'class': 'OggZipDataset',
            'path': path,
            'segment_file': training_file,
            'partition_epoch': 1,
            'use_cache_manager': True,
            'audio': audio_opts,
            'targets': None,
            'seq_ordering': "laplace:.1000" if shuffle else "sorted",
        }
    dataset = init_dataset(d)
    return dataset

mel_basis = {}
hann_window = {}


def extract_features(feature_name, audio, sr, win_size, hop_size, num_ff, f_min, f_max, num_mels, center,
                     min_amp):
    # calculate padding length and pad audio
    pad_hop = (sr * hop_size)
    pad_ff = (sr * win_size)
    audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0)
    audio = torch.nn.functional.pad(audio, (int((pad_ff-pad_hop)/2), int((pad_ff-pad_hop)/2)), mode='reflect')
    audio = audio.squeeze(0).squeeze(0)
    audio = np.array(audio)
    
    if feature_name == "mfcc":
        feature_data = _get_audio_features_mfcc(audio, sr, win_size, hop_size, num_ff, num_mels, f_min, f_max, center)
    elif feature_name == "log_mel_filterbank":
        feature_data = _get_audio_log_mel_filterbank(audio, sr, win_size, hop_size, num_ff)
    elif feature_name == "log_log_mel_filterbank":
        feature_data = _get_audio_log_log_mel_filterbank(audio, sr, win_size, hop_size, num_ff)
    elif feature_name == "db_mel_filterbank":
        feature_data = _get_audio_db_mel_filterbank(audio, sr, win_size, hop_size, num_ff, f_min, f_max, min_amp, center)
    elif feature_name == "linear_spectrogram":
        feature_data = _get_audio_linear_spectrogram(audio, sr, win_size, hop_size, num_ff, center)
    return feature_data


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, audio_form, training_files, segment_size, num_ff,
                 hop_size, win_size, sampling_rate,  fmin=60, fmax=7600, split=True, shuffle=True,
                 n_cache_reuse=1, device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None,
                 features="db_mel_filterbank", center=False, num_mels=128, min_amp=1e-10, path=None):
        self.audio_files = training_files
        random.seed(1234)
        self.shuffle = shuffle
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.audio_form = audio_form
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
        self.min_amp = min_amp
        self.path = path
        self.dataset = get_dataset(self.path, self.audio_files, self.shuffle)

    def __getitem__(self, index):
        # caching done in ogg_zip
        # need seperate locks for loading and getting data
        load_lock.acquire()
        load_seqs(self.dataset, index)
        load_lock.release()
        data_lock.acquire()
        audio, audio_tag = get_audio(index, self.dataset)
        data_lock.release()
        # only normalize with MAX_WAV if we have .wav audio format 
        if self.audio_form == ".wav":
            audio = audio / MAX_WAV_VALUE
        if not self.fine_tuning:
            audio = normalize(audio) * 0.95

        audio = torch.FloatTensor(audio)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
            
            logging.error(np.shape(audio))
            # compute mels for training
            mel = extract_features(self.features, np.array(audio.squeeze()), self.sampling_rate, self.win_size,
                                   self.hop_size, self.num_ff, self.fmin, self.fmax, self.num_mels, self.center,
                                   self.min_amp)
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
                                    self.hop_size, self.num_ff, self.fmin, self.fmax_loss, self.num_mels, self.center,
                                    self.min_amp)
        mel_loss = np.swapaxes(mel_loss, 0, 1)
        mel_loss = np.expand_dims(mel_loss, axis=0)

        # compute noise based on mel shape for generation of fake audio
        noise = torch.randn([64, mel.shape[-1]])
        return (mel.squeeze(), audio.squeeze(0), audio_tag , mel_loss.squeeze(), noise)

    def __len__(self):
        return self.dataset.get_total_num_seqs()
