import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import read_wav_np, load_audio
import sys
sys.path.append("/u/schuemann/experiments/tts_asr_2021/recipe/returnn_new")
from returnn.datasets.util.feature_extraction import ExtractAudioFeatures


def get_dataset_filelist(hp, tmpdir):
    with open(hp.data.train_files, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(tmpdir, x.split('|')[0] + hp.audio.audio_form)
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(hp.data.validation_files, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(tmpdir, x.split('|')[0] + hp.audio.audio_form)
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files

def create_dataloader(hp, args, train, validation_file, training_file):

    train_dataset = MelFromDisk(hp, args, train, file_list=training_file)
    valid_dataset = MelFromDisk(hp, args, train, file_list=validation_file)

    if train:
        return DataLoader(dataset=train_dataset, batch_size=hp.train.batch_size, shuffle=True,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=True)
    else:
        return DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False, drop_last=False)

def extract_features(feature_name, audio, sr, win_size, hop_size, num_ff, f_min, f_max, num_mels, center,
                     min_amp, with_delta=False, norm_mean=None, norm_std_dev=None, random_permute=None,
                     random_state=None, raw_ogg_opts=None, pre_process=None, post_process=None, num_channels=None,
                     peak_norm=True, preemphasis=None, join_frames=None):

    # calculate padding length and pad audio such that features fit to audio in training/validation
    # different padding based on filter length in hifigan
    pad_ff = (sr * win_size)

    # make tensor 3D for padding
    audio = torch.tensor(audio).unsqueeze(1)
    print('shape in feature extraction')
    print(np.shape(audio))
    if len(np.shape(audio))<3:
        audio = np.swapaxes(audio, 1, 0)
        audio = audio.unsqueeze(0)
    audio = torch.nn.functional.pad(audio, (int((pad_ff)/2), int((pad_ff)/2)), mode='reflect')

    # make audio size suitable for feature extraction in Returnn
    audio = audio.squeeze(0).squeeze(0)
    audio = np.array(audio)
    print(np.shape(audio))
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
    print(np.shape(feature_data))
    return feature_data

class MelFromDisk(Dataset):
    def __init__(self, hp, args, train, file_list):
        self.hp = hp
        self.args = args
        self.train = train
        self.hop_length = int(hp.audio.step_length * hp.audio.sampling_rate)
        self.mel_segment_length = int(hp.audio.segment_length // self.hop_length + 2)
        self.files = file_list
        self.mapping = [i for i in range(len(self.files))]


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        if self.train:
            # why choose two and why is first not random on shuffling?
            idx1 = idx
            idx2 = self.mapping[idx1]
            return self.my_getitem(idx1), self.my_getitem(idx2)
        else:
            return self.my_getitem(idx)

    def shuffle_mapping(self):
        random.shuffle(self.mapping)

    def my_getitem(self, idx):
        audiopath = self.files[idx]

        # mel_path = "{}/{}.npy".format(self.hp.data.mel_path, id)
        audio, sr = load_audio(audiopath)
        if self.hp.audio.audio_form == ".wav":
            audio = audio/32768.0
        if len(audio) < self.hp.audio.segment_length + self.hp.audio.pad_short:
            audio = np.pad(audio, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio)), \
                    mode='constant', constant_values=0.0)

        audio = torch.from_numpy(audio).unsqueeze(0)
        # mel = torch.load(melpath).squeeze(0) # # [num_mel, T]
        print(np.shape(audio))
        # mel = torch.from_numpy(np.load(mel_path))
        mel = extract_features(self.hp.audio.features, np.array(audio), self.hp.audio.sampling_rate,
                               self.hp.audio.win_length, self.hp.audio.step_length, self.hp.audio.number_feature_filters,
                               self.hp.audio.mel_fmin, self.hp.audio.mel_fmax, self.hp.audio.num_mels,
                               self.hp.audio.center, self.hp.audio.min_amp, self.hp.audio.with_delta, self.hp.audio.norm_mean, self.hp.audio.norm_std_dev,
                               self.hp.audio.random_permute, self.hp.audio.random_state, self.hp.audio.raw_ogg_opts,
                               self.hp.audio.pre_process, self.hp.audio.post_process, self.hp.audio.num_channels,
                               self.hp.audio.peak_norm, self.hp.audio.preemphasis, self.hp.audio.join_frames)
        mel = np.swapaxes(mel, 0, 1) 
        print("mel shape in dataloader")
        print(np.shape(mel)) # needs to be 80 x something
        mel = torch.tensor(mel)        
        # chooses random length of mel for training ? possibly to generate some variation? -> audio is fitted to mel
        # hop_size = step_len * sampling_rate ?
        if self.train:
            max_mel_start = mel.size(1) - self.mel_segment_length
            mel_start = random.randint(0, max_mel_start)
            mel_end = int(mel_start + self.mel_segment_length)
            mel = mel[:, mel_start:mel_end]

            audio_start = mel_start * self.hop_length
            audio = audio[:, audio_start:audio_start+self.hp.audio.segment_length]

        if self.hp.audio.audio_form == ".wav":
            audio = audio + (1/32768)
        audio = audio * torch.randn_like(audio)
        audio = audio.float()
        return mel, audio
