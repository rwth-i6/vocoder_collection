# UnivNet
[UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation.](https://arxiv.org/abs/2106.07889)

## Training
```
python train.py --config config_c32.json
```
## Citation
```
@misc{seo2021controlling,
      title={Controlling Neural Networks with Rule Representations}, 
      author={Sungyong Seo and Sercan O. Arik and Jinsung Yoon and Xiang Zhang and Kihyuk Sohn and Tomas Pfister},
      year={2021},
      eprint={2106.07804},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Note
* For more complete and end to end Voice cloning or Text to Speech (TTS) toolbox 🧰 please visit [Deepsync Technologies](https://deepsync.co/).

## References:
* [Hi-Fi-GAN repo](https://github.com/jik876/hifi-gan)
* [LVCNet repo](https://github.com/ZENGZHEN-TTS/LVCNet)

# HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis 

Unofficial PyTorch implementation of [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646).
<br>**HiFi-GAN :**
![](./assets/fig1.JPG)
<br>

![](./assets/fig2.JPG)
<br>

![](./assets/fig3.JPG)


## Note
* For more complete and end to end Voice cloning or Text to Speech (TTS) toolbox 🧰 please visit [Deepsync Technologies](https://deepsync.co/).

## Prerequisites

Tested on Python 3.6
```bash
pip install -r requirements.txt
```

## Prepare Dataset

- Download dataset for training. This can be any wav files with sample rate 22050Hz. (e.g. LJSpeech was used in paper)
- preprocess: `python preprocess.py -c config/default.yaml -d [data's root path]`
- Edit configuration `yaml` file

## Train & Tensorboard

- `python trainer.py -c [config yaml file] -n [name of the run]`
  - `cp config/default.yaml config/config.yaml` and then edit `config.yaml`
  - Write down the root path of train/validation files to 2nd/3rd line.
  - Each path should contain pairs of `*.wav` with corresponding (preprocessed) `*.mel` file.
  - The data loader parses list of files within the path recursively.
- `tensorboard --logdir logs/`

## Pretrained model
Check [here](https://drive.google.com/drive/folders/1YpDbMpa2i1MLsJ7-qy9o3JOG8ZCDKWRo?usp=sharing).

## Inference

- `python inference.py -p [checkpoint path] -i [input mel path]`

# correct training and inference execution for both univnet and hifigan still needs to be updated in the README

