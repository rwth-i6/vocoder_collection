from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import sys
import numpy as np
import os
import argparse
import json
from scipy.io.wavfile import read
import torch
import soundfile as sf
from librosa.util import normalize
from scipy.io.wavfile import write
from meldataset import MAX_WAV_VALUE
from generator import UnivNet as Generator
from utils import HParam, AttrDict, build_env
from meldataset import extract_features
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


def inference(a, h, with_postnet=False):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            audio, sr = sf.read(os.path.join(a.input_wavs_dir, filename))
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            audio = torch.FloatTensor(audio)
            audio = np.array(audio)
            mel = extract_features(a.features, audio, h.sampling_rate, h.win_size, h.hop_size, h.num_ff,
                                   h.fmin, h.fmax_for_loss, h.num_mels, h.center, h.min_amp)
            mel = np.expand_dims(mel, axis=0)
            mel = np.swapaxes(mel, 1, 2)
            mel = torch.tensor(mel)
            noise = torch.randn([1, 64, mel.shape[-1]])
            audio = generator.forward(noise, mel)
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
