import itertools
import numpy as np
import os
import time
import tempfile
import shutil
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
from meldataset import MelDataset, get_dataset_filelist, extract_features
from generator import UnivNet
from discriminator import MultiPeriodDiscriminator, MultiResSpecDiscriminator
from loss import feature_loss, generator_loss, discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint, load_normal_data
from stft_loss import MultiResolutionSTFTLoss
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

torch.backends.cudnn.benchmark = True


def train(rank, a, h):
    # assert that segment lengths fit after upsampling, otherwise there is an incorrect USR to feature length ratio
    feature_length = (h.segment_size//(h.hop_size*h.sampling_rate))
    sr = np.prod(h.upsample_rates)
    assert feature_length * sr == h.segment_size, "incorrect upsampling rate to feature length ratio"

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

    # create temporary directory and copy audio files    
    tmpdir = tempfile.TemporaryDirectory()
    shutil.copytree(a.input_audio_dir, tmpdir.name, dirs_exist_ok=True)

    training_filelist, validation_filelist = get_dataset_filelist(a, tmpdir.name)
    
    if a.hdf_train and a.hdf_val:
        hdf_seq_t, hdf_tag_t = load_normal_data(a.hdf_train)   
        hdf_seq_val, hdf_tag_val = load_normal_data(a.hdf_val)
    else:
        print("no hdf for training/validation available")
    trainset = MelDataset(hdf_seq_t, hdf_tag_t, a.target_audio_form, training_filelist, h.segment_size, h.num_ff,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, split=True, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          base_mels_path=a.input_mels_dir, features=a.features,
                          num_mels=h.num_mels, center=h.center, min_amp=h.min_amp, with_delta=h.with_delta,
                          norm_mean=h.norm_mean, norm_std_dev=h.norm_std_dev, random_permute=h.random_permute,
                          random_state=h.random_state, raw_ogg_opts=h.raw_ogg_opts, pre_process=h.pre_process,
                          post_process=h.post_process, num_channels=h.num_channels, peak_norm=h.peak_norm,
                          preemphasis=h.preemphasis, join_frames=h.join_frames)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
    
    if rank == 0:
        validset = MelDataset(hdf_seq_val, hdf_tag_val, a.target_audio_form, validation_filelist, h.segment_size, h.num_ff,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device,
                              base_mels_path=a.input_mels_dir, features=a.features, num_mels=h.num_mels,
                              center=h.center, min_amp=h.min_amp, with_delta=h.with_delta,
                              norm_mean=h.norm_mean, norm_std_dev=h.norm_std_dev, random_permute=h.random_permute,
                              random_state=h.random_state, raw_ogg_opts=h.raw_ogg_opts, pre_process=h.pre_process,
                              post_process=h.post_process, num_channels=h.num_channels, peak_norm=h.peak_norm,
                              preemphasis=h.preemphasis, join_frames=h.join_frames)

        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    stft_loss = MultiResolutionSTFTLoss()

    print("Training with {} features".format(a.features))

    for epoch in range(max(0, last_epoch), a.training_epochs):
        print("current rank:", rank)
        print("current process:", os.getpid())
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
            
            y_g_hat_mel = [] 
            for audio in y_g_hat.squeeze(1).detach().cpu().numpy():
                single_mel = extract_features(a.features, audio, h.sampling_rate, h.win_size, h.hop_size, h.num_ff,
                                              h.fmin, h.fmax_for_loss, h.num_mels, h.center, h.min_amp, h.with_delta,
                                              h.norm_mean, h.norm_std_dev, h.random_permute, h.random_state,
                                              h.raw_ogg_opts, h.pre_process, h.post_process, h.num_channels,
                                              h.peak_norm, h.preemphasis, h.join_frames)
                y_g_hat_mel.append(single_mel)
            
            y_g_hat_mel = torch.tensor(y_g_hat_mel)
            y_g_hat_mel = np.swapaxes(y_g_hat_mel, 1, 2)
            
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

            sc_loss, mag_loss = stft_loss(y_g_hat[:, :, :y.size(2)].squeeze(1), y.squeeze(1))

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
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

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
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = []
                            for audio in y_g_hat.squeeze(1).detach().cpu().numpy():
                                single_mel = extract_features(a.features, audio, h.sampling_rate, h.win_size,
                                                              h.hop_size, h.num_ff, h.fmin, h.fmax_for_loss, h.num_mels,
                                                              h.center, h.min_amp, h.with_delta, h.norm_mean,
                                                              h.norm_std_dev, h.random_permute, h.random_state,
                                                              h.raw_ogg_opts, h.pre_process, h.post_process,
                                                              h.num_channels, h.peak_norm, h.preemphasis,
                                                              h.join_frames)

                                y_g_hat_mel.append(single_mel)
                            y_g_hat_mel = torch.tensor(np.swapaxes(y_g_hat_mel, 1, 2))
                            y_g_hat_mel = y_g_hat_mel.to("cuda:0")
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                               
                                y_hat_spec = []
                                for audio in y_g_hat.squeeze(1).detach().cpu().numpy():
                                    single_mel = extract_features(a.features, audio, h.sampling_rate, h.win_size,
                                                                  h.hop_size, h.num_ff, h.fmin, h.fmax_for_loss,
                                                                  h.num_mels, h.center, h.min_amp, h.with_delta,
                                                                  h.norm_mean, h.norm_std_dev, h.random_permute,
                                                                  h.random_state, h.raw_ogg_opts, h.pre_process,
                                                                  h.post_process, h.num_channels, h.peak_norm,
                                                                  h.preemphasis, h.join_frames)
                                    y_hat_spec.append(single_mel)
                                y_hat_spec = torch.tensor(np.swapaxes(y_hat_spec, 1, 2))
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
    parser.add_argument('--input_audio_dir', default=None, help='path to audio directory for training/validation data')
    parser.add_argument('--input_mels_dir', default=None, help='path to dataset for finetuning data') 
    parser.add_argument('--input_training_file', default=None,
                        help='path to LJSpeech training file')
    parser.add_argument('--input_validation_file', default=None,
                        help='path to LJSpeech validation  file')
    parser.add_argument('--checkpoint_path', default=None, help='path to where checkpoints should be saved') 
    parser.add_argument('--config', default='config_univ.json', help='path to config file')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=15000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--features', default='db_mel_filterbank',
                        help='choose features from "mfcc", "log_mel_filterbank", "log_log_mel_filterbank", '
                             '"db_mel_filterbank", "linear_spectrogram"')
    parser.add_argument('--target_audio_form', default='.ogg', help="choose target audio representation to load the "
                                                                    "audio data for training (.wav, .ogg...)")
    parser.add_argument('--hdf_train', help="path to hdf for vocoder training with hdf instead of audio files")
    parser.add_argument('--hdf_val', help="path to hdf for vocoder training with hdf instead of audio files")

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config_univ.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        print('CUDA device count:', torch.cuda.device_count())
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
