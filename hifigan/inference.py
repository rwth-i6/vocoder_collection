import os
import soundfile as sf
import subprocess
import glob
import h5py
import torch
import argparse
from scipy.io.wavfile import write
import numpy as np
from model.generator import Generator
from utils.hparams import HParam, load_hparam_str
from denoiser import Denoiser
from datasets.dataloader import extract_features
MAX_WAV_VALUE = 32768.0

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

def save_ogg(wav, path, sr):
    """

    :param wav:
    :param path:
    :param sr:
    :return:
    """
    p1 = subprocess.Popen(["ffmpeg", "-y", "-f", "s16le", "-ar", "%i" % sr, "-i", "pipe:0", path],
                          stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE)
    p1.communicate(input=wav.astype(np.int16).tobytes())
    p1.terminate()

def main(args):
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])

    model = Generator(hp.audio.number_feature_filters)

    model.load_state_dict(checkpoint['model_g'])
    model.eval(inference=True)
    hop_length_audio = int(hp.audio.step_length * hp.audio.sampling_rate)
    if args.hdf:
        sequences, tags = load_normal_data(args.hdf)
        with torch.no_grad():
            for mel, tag in zip(sequences, tags):
                mel = np.expand_dims(mel, axis=0)
                mel = np.swapaxes(mel, 1, 2)
                mel = torch.tensor(mel)
                audio = model.inference(mel)
                audio = audio.squeeze()
                audio = audio[:-(hop_length_audio*10)]
                audio = MAX_WAV_VALUE * audio
                audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
                audio = audio.short()
                audio = audio.cpu().detach().numpy()
                print(np.shape(audio))
                print(len(audio))
                segment_length = float(len(audio))/float(hp.audio.sampling_rate)
                output_file = os.path.join(
                    args.output_dir,
                    (str(segment_length) + "_" + tag + args.audio_form).replace("/","_")
                )
                if args.audio_form == ".wav":
                    write(output_file, hp.audio.sampling_rate, audio)
                else:
                    save_ogg(audio, output_file, hp.audio.sampling_rate)
                print(output_file)
   
    else:
        filelist = os.listdir(args.input)
        with torch.no_grad():
            for filename in filelist:
                audio, sr = sf.read(os.path.join(args.input, filename))
                mel = extract_features(hp.audio.features, audio,
                                                           hp.audio.sampling_rate, hp.audio.win_length, hp.audio.step_length,
                                                           hp.audio.number_feature_filters, hp.audio.mel_fmin, hp.audio.mel_fmax,
                                                           hp.audio.num_mels, hp.audio.center, hp.audio.min_amp,
                                                           hp.audio.with_delta, hp.audio.norm_mean, hp.audio.norm_std_dev,
                                                           hp.audio.random_permute, hp.audio.random_state, hp.audio.raw_ogg_opts,
                                                           hp.audio.pre_process, hp.audio.post_process, hp.audio.num_channels,
                                                           hp.audio.peak_norm, hp.audio.preemphasis, hp.audio.join_frames) 
                mel = np.expand_dims(mel, axis=0)
                mel = np.swapaxes(mel, 1, 2)
                print(np.shape(mel))
                mel = torch.tensor(mel)
                audio = model.inference(mel)
                audio = audio.squeeze(0) 
                if args.d:
                    denoiser = Denoiser(model).cuda()
                    audio = denoiser(audio, 0.1)
                audio = audio.squeeze()
                audio = MAX_WAV_VALUE * audio
                audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
                audio = audio.short()
                audio = audio.cpu().detach().numpy()
                output_file = os.path.join(
                    args.output_dir,
                    os.path.splitext(filename)[0] + args.audio_form
                )
                write(output_file, hp.audio.sampling_rate, audio)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input', type=str,
                        help="directory of audio for testing inference ")
    parser.add_argument('-d', action='store_true', help="denoising ")
    parser.add_argument('-audio_form', default=".ogg")
    parser.add_argument('-o', "--output_dir", default=None)
    parser.add_argument('-s', "--hdf", default=None, help="path to hdf file for inference")
    args = parser.parse_args()

    main(args)
