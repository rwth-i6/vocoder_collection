from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import sys
import numpy as np
import os
import h5py
import argparse
import json
from scipy.io.wavfile import read
import torch
import soundfile as sf
from librosa.util import normalize
from scipy.io.wavfile import write
from meldataset import MAX_WAV_VALUE, extract_features
from generator import UnivNet as Generator
from utils import HParam, AttrDict, build_env
sys.path.append("/u/schuemann/experiments/tts_asr_2021/recipe/returnn_new")
h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

# load hdf data
def load_normal_data(hdf):
    input_data = h5py.File(hdf, "r")
    num_seqs = -1
    inputs = input_data['inputs']
    seq_tags = input_data['seqTags']
    lengths = input_data['seqLengths']
    sizes = None

    sequences = []
    tags = []
    offset = 0
    for tag, length in zip(seq_tags, lengths):
        tag = tag if isinstance(tag, str) else tag.decode()
        in_data = inputs[offset:offset + length[0]]
        sequences.append(in_data)
        offset += length[0]
        tags.append(tag)
        if len(sequences) == num_seqs:
            break
    return sequences, tags

def inference(a, h, with_postnet=False):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()

    # Generate audio from mel spectrograms saved in hdf file. 
    # The generated filename will be the tag of the sequence. (only works correctly for librispeech as of now)
    
    if a.hdf:
        sequences, tags = load_normal_data(a.hdf)
        with torch.no_grad():
            for mel, tag in zip(sequences, tags):
                mel = np.expand_dims(mel, axis=0)
                mel = np.swapaxes(mel, 1, 2)
                mel = torch.tensor(mel)
                noise = torch.randn([1, 64, mel.shape[-1]])
                audio = generator.forward(noise, mel)
                #if a.audio_form == ".wav":
                #    audio = audio * MAX_WAV_VALUE
                audio = audio.cpu().numpy().astype('int16')
                output_file = os.path.join(
                    a.output_dir,
                    tag.split("/")[-1] + a.audio_form
                )
                write(output_file, h.sampling_rate, audio)
                print(output_file)

    # Test audio generation by providing path to audio file
    # Audio path -> mel spectrogram -> generated audio
    else:
        filelist = os.listdir(a.input_wavs_dir)
        with torch.no_grad():
            for i, filename in enumerate(filelist):
                audio, sr = sf.read(os.path.join(a.input_wavs_dir, filename))
                if a.audio_form == ".wav":
                    audio = audio / MAX_WAV_VALUE
                audio = normalize(audio) * 0.95
                audio = torch.FloatTensor(audio)
                audio = np.array(audio)

                mel = extract_features(a.features, audio, h.sampling_rate, h.win_size, h.hop_size, h.num_ff, h.fmin,
                                       h.fmax_for_loss, h.num_mels, h.center, h.min_amp, h.with_delta, h.norm_mean,
                                       h.norm_std_dev, h.random_permute, h.random_state, h.raw_ogg_opts, h.pre_process,
                                       h.post_process, h.num_channels, h.peak_norm, h.preemphasis, h.join_frames)
                mel = np.expand_dims(mel, axis=0)
                mel = np.swapaxes(mel, 1, 2)
                mel = torch.tensor(mel)
                noise = torch.randn([1, 64, mel.shape[-1]])
                audio = generator.forward(noise, mel)
                if a.audio_form == ".wav":
                    audio = audio * MAX_WAV_VALUE
                audio = audio.cpu().numpy().astype('int16')
                output_file = os.path.join(
                    a.output_dir,
                    os.path.splitext(filename)[0] + a.audio_form
                )
                write(output_file, h.sampling_rate, audio)
                print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--config', default='config_univ.json')
    parser.add_argument('--features', default='db_mel_filterbank',
                        help='choose features from "mfcc", "log_mel_filterbank", "log_log_mel_filterbank", '
                             '"db_mel_filterbank", "linear_spectrogram"')
    parser.add_argument('--audio_form', default='.wav', help="choose audio representation to safe the "
                                                             "audio data (wav,ogg...)")
    parser.add_argument('--hdf', default=None, help="choose hdf as mel input instead")
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
   
    torch.manual_seed(h.seed)
    global device
    device = torch.device('cpu')

    inference(args, h)


if __name__ == '__main__':
    main()
