from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, BandPassFilter, AddGaussianSNR
import numpy as np
from scipy.io.wavfile import write
import torch
import os
from dataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from librosa.util import normalize
def main():
	augment = Compose([
			AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
			TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
			AddGaussianSNR(min_snr_in_db=10.0, max_snr_in_db=30.0, p=1.0),
			BandPassFilter(min_center_freq=200, max_center_freq=4000),
			Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
	])

	filelist = os.listdir("data/clean")
	for i, filename in enumerate(filelist):
		wav, sr = load_wav(os.path.join("data/clean", filename))
		wav = wav / MAX_WAV_VALUE
		wav = normalize(wav) * 0.95
		wav = torch.FloatTensor(wav)
		output_file = os.path.join(
				"data/wavs",
				os.path.splitext(filename)[0] + '.wav'
				)

		augmented_samples = augment(samples=np.array(wav), sample_rate=sr)
		write(output_file, sr, np.array(augmented_samples))
		print(output_file)

if __name__ == "__main__":
    main()

