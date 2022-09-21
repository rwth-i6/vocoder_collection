from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import numpy as np
import os
import argparse
import json
import torch
from librosa.util import normalize
from scipy.io.wavfile import write
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from generator import UnivNet as Generator
from utils import HParam, AttrDict, build_env
from returnn.datasets.util.feature_extraction import _get_audio_db_mel_filterbank
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


def inference(a, h, with_postnet=False):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filename))
            wav = wav / MAX_WAV_VALUE
            wav = normalize(wav) * 0.95
            wav = torch.FloatTensor(wav)
            wav = np.array(wav)
            mel = _get_audio_db_mel_filterbank(wav, h.sampling_rate, h.win_size, h.hop_size, h.num_mels,
                                                           h.fmin, h.fmax_for_loss)
            mel = np.expand_dims(mel,axis=0)
            mel = np.swapaxes(mel,1,2)
            mel = torch.tensor(mel)
            noise = torch.randn([1,64, mel.shape[-1]]) 
            audio = generator.forward(noise, mel)
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            output_file = os.path.join(
                a.output_dir,
                os.path.splitext(filename)[0] + '_generated.wav'
            )
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--config', default='config_8200.json')

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
