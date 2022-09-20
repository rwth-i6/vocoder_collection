import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import numpy as np
import h5py
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from utils import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from generator import UnivNet
from discriminator import MultiPeriodDiscriminator, MultiResSpecDiscriminator
from loss import feature_loss, generator_loss, discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from stft_loss import MultiResolutionSTFTLoss
import sys
sys.path.append("/u/schuemann/experiments/tts_asr_2021/recipe/")
from returnn.datasets.util.feature_extraction import _get_audio_db_mel_filterbank
torch.backends.cudnn.benchmark = True

def load_normal_data():
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

def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = UnivNet(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiResSpecDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    # load hdf features
    hdf_input_data = h5py.File(
        "/u/schuemann/experiments/tts_asr_2021/work/i6_private/users/schuemann/tts/hdf/ReturnnDumpHDFJob.ZMmXSWlaIvvb/output/data.hdf",
        'r')
    num_seqs = -1
    hdf_inputs = hdf_input_data['inputs']
    hdf_seq_tags = hdf_input_data['seqTags']
    hdf_lengths = hdf_input_data['seqLengths']
    sizes = None
    if hdf_input_data.get('targets', {}).get('data', {}).get('sizes', None):
        sizes = np.reshape(input_data['targets']['data']['sizes'], (-1, 2))

    hdf_sequences = []
    hdf_tags = []
    offset = 0
    for tag, length in zip(hdf_seq_tags, hdf_lengths):
        tag = tag if isinstance(tag, str) else tag.decode()
        in_data = hdf_inputs[offset:offset + length[0]]
        hdf_sequences.append(in_data)
        offset += length[0]
        hdf_tags.append(tag)
        if len(hdf_sequences) == num_seqs:
            break

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, split=True, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir, hdf_sequences=hdf_sequences, hdf_tags=hdf_tags)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
    
    # also split audio data for vailidation
    if rank == 0:
        #validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
        #                      h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
        #                      fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
        #                      base_mels_path=a.input_mels_dir, hdf_sequences=hdf_sequences, hdf_tags=hdf_tags)
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir, hdf_sequences=hdf_sequences, hdf_tags=hdf_tags)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=True,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    stft_loss = MultiResolutionSTFTLoss()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel, z = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            z = torch.autograd.Variable(z.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(z, x)
            #print("fake audio:")
            #print(np.shape(y_g_hat))
            #y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, 256,
            #                              1024,
            #                              h.fmin, h.fmax_for_loss)
            
            # need to pull tensor to cpu and then to gpu again: inefficient?
            # might need to cut audio at the end due to not matching segment size
            # approximate predicted fake audio as otherwise we cannot evaluate on full sequences in validation
            y_g_hat_mel = [] 
            for audio in y_g_hat.squeeze(1).detach().cpu().numpy():
                single_mel = _get_audio_db_mel_filterbank(audio, h.sampling_rate, h.win_size, h.hop_size, h.num_mels,
                                                           h.fmin, h.fmax_for_loss)
                y_g_hat_mel.append(single_mel)
            
            y_g_hat_mel = torch.tensor(y_g_hat_mel)
            print(np.shape(y_g_hat_mel))
            print(np.shape(y))
            y_g_hat_mel = np.swapaxes(y_g_hat_mel,1,2)
            # pull to gpu
            y_g_hat_mel = y_g_hat_mel.to("cuda:0")
            if steps > h.disc_start_step:

                optim_d.zero_grad()

                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_disc_all = loss_disc_s + loss_disc_f

                loss_disc_all.backward()
                optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            #loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
           
            sc_loss, mag_loss = stft_loss(y_g_hat.squeeze(1), y[:, :, :y_g_hat.size(2)].squeeze(1))
            #sc_loss, mag_loss = stft_loss(torch.nn.functional.pad(y_g_hat,(0,y.size(2),0,0,0,0))[:,:,:y.size(2)].squeeze(1), y.squeeze(1))

            loss_mel = h.lambda_aux * (sc_loss + mag_loss)  # STFT Loss

            if steps > h.disc_start_step:
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            else:
                loss_gen_all = loss_mel

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel[:,:,:y_mel.size(2)]).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                             else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                             else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel, z = batch
                            y_g_hat = generator(z.to(device), x.to(device))
                            #print("input mel")
                            #print(np.shape(x.to(device)))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            real_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          256, 1024,
                                                          h.fmin, h.fmax_for_loss)
                            #print("fake audio:")
                            #print(np.shape(y_g_hat.squeeze(1).detach().cpu().numpy()))
                            #print(np.shape(y_g_hat.squeeze(1)[:,:y.size(1)].detach().cpu().numpy()))
                            #print("actual mel with fake audio")
                            #print(np.shape(real_mel))
                            #print("real audio")
                            #print(np.shape(y))
                            #print("my real mel")
                            #print(np.shape(y_mel))
                            #print("my fake mel")
                            
                            y_g_hat_mel = []
                            for audio in y_g_hat.squeeze(1).detach().cpu().numpy():
                                single_mel = _get_audio_db_mel_filterbank(audio, h.sampling_rate, h.win_size, h.hop_size, h.num_mels,
                                                                          h.fmin, h.fmax_for_loss)
                                y_g_hat_mel.append(single_mel)
                            y_g_hat_mel = torch.tensor(np.swapaxes(y_g_hat_mel,1,2))
                            print(np.shape(y_g_hat_mel))
                            print(np.shape(y))
                            y_g_hat_mel = y_g_hat_mel.to("cuda:0")
                            val_err_tot += F.l1_loss(y_mel[:,:,:y_g_hat_mel.size(2)], y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                #y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                #                             h.sampling_rate, h.hop_size, h.win_size,
                                #                             h.fmin, h.fmax)
                               
                                y_hat_spec = []
                                for audio in y_g_hat.squeeze(1).detach().cpu().numpy():
                                    single_mel = _get_audio_db_mel_filterbank(audio, h.sampling_rate, h.win_size, h.hop_size, h.num_mels,
                                                                              h.fmin, h.fmax_for_loss)
                                    y_hat_spec.append(single_mel)
                                y_hat_spec = torch.tensor(np.swapaxes(y_hat_spec,1,2))
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='/work/asr3/rossenbach/schuemann/vocoder/UnivNet-pytorch-test/LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='/work/asr3/rossenbach/schuemann/vocoder/UnivNet-pytorch-test-full/ft_dataset')
    parser.add_argument('--input_training_file', default='/work/asr3/rossenbach/schuemann/vocoder/UnivNet-pytorch-test-full/LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='/work/asr3/rossenbach/schuemann/vocoder/UnivNet-pytorch-test-full/LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='/work/asr3/rossenbach/schuemann/vocoder/UnivNet-pytorch-test-full/cp_hifigan')
    parser.add_argument('--config', default='config_c32.json')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config_c32.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
