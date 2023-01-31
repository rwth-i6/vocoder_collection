import math
import os
import random
import torch
import torch.utils.data
import soundfile as sf
import numpy as np
from librosa.util import normalize
import sys
sys.path.append("/u/schuemann/experiments/tts_asr_2021/recipe/returnn_new")
from returnn.datasets.util.feature_extraction import ExtractAudioFeatures
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


def extract_features(feature_name, audio, sr, win_size, hop_size, num_ff, f_min, f_max, num_mels, center,
                     min_amp, with_delta=False, norm_mean=None, norm_std_dev=None, random_permute=None,
                     random_state=None, raw_ogg_opts=None, pre_process=None, post_process=None, num_channels=None,
                     peak_norm=True, preemphasis=None, join_frames=None):

    # calculate padding length and pad audio such that features fit to audio in training/validation
    pad_hop = (sr * hop_size)
    pad_ff = (sr * win_size)
    audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0)
    audio = torch.nn.functional.pad(audio, (int((pad_ff-pad_hop)/2), int((pad_ff-pad_hop)/2)), mode='reflect')
    audio = audio.squeeze(0).squeeze(0)
    audio = np.array(audio)

    # add extra feature options and calculate features, extra feature options that are not wanted need to be kept as default option None
    feature_options = {}
    if center is not None:
        feature_options.update({"center": center})
    if min_amp is not None:
        feature_options.update({"min_amp": min_amp})
    if f_min is not None:
        feature_options.update({"fmin": f_min})
    if f_max is not None:
        feature_options.update({"fmax": f_max})
    if num_mels is not None:
        feature_options.update({"n_mels": num_mels})   

    Features = ExtractAudioFeatures(win_size, hop_size, num_ff, with_delta, norm_mean, norm_std_dev, feature_name,
                                    feature_options, random_permute, random_state, raw_ogg_opts,
                                    pre_process, post_process, sr, num_channels, peak_norm, preemphasis, join_frames)
    feature_data = Features.get_audio_features(audio, sr)
    return feature_data


def get_dataset_filelist(a, audio_form):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_audio_dir, x.split('|')[0] + audio_form)
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_audio_dir, x.split('|')[0] + audio_form)
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, audio_form, training_files, segment_size, num_ff,
                 hop_size, win_size, sampling_rate,  fmin=60, fmax=7600, split=True, shuffle=True,
                 n_cache_reuse=1, device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None,
                 features="db_mel_filterbank", center=False, num_mels=128, min_amp=1e-10, with_delta=False,
                 norm_mean=None, norm_std_dev=None, random_permute=None, random_state=None, raw_ogg_opts=None,
                 pre_process=None, post_process=None, num_channels=None, peak_norm=True, preemphasis=None,
                 join_frames=None):

        self.audio_files = training_files
        # shuffle audio based on random seed if shuffle=True
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
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
        self.with_delta = with_delta
        self.norm_mean = norm_mean
        self.norm_std_dev = norm_std_dev
        self.random_permute = random_permute
        self.random_state = random_state
        self.raw_ogg_opts = raw_ogg_opts
        self.pre_process = pre_process
        self.post_process = post_process
        self.num_channels = num_channels
        self.peak_norm = peak_norm
        self.preemphasis = preemphasis
        self.join_frames = join_frames

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_audio(filename)
            if self.audio_form == ".wav":
                audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
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
                                   self.hop_size, self.num_ff, self.fmin, self.fmax, self.num_mels, self.center,
                                   self.min_amp, self.with_delta, self.norm_mean, self.norm_std_dev,
                                   self.random_permute, self.random_state, self.raw_ogg_opts, self.pre_process,
                                   self.post_process, self.num_channels, self.peak_norm, self.preemphasis,
                                   self.join_frames)
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
                                    self.min_amp, self.with_delta, self.norm_mean, self.norm_std_dev,
                                    self.random_permute, self.random_state, self.raw_ogg_opts, self.pre_process,
                                    self.post_process, self.num_channels, self.peak_norm, self.preemphasis,
                                    self.join_frames)

        mel_loss = np.swapaxes(mel_loss, 0, 1)
        mel_loss = np.expand_dims(mel_loss, axis=0)

        # compute noise based on mel shape for generation of fake audio
        noise = torch.randn([64, mel.shape[-1]])
        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze(), noise)

    def __len__(self):
        return len(self.audio_files)
