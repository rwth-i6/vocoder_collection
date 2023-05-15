import os
import time
import logging
import argparse
import tempfile
import shutil
from utils.train import train
from utils.hparams import HParam
from utils.writer import MyWriter
from datasets.dataloader import create_dataloader, get_dataset_filelist, load_normal_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    parser.add_argument('-t', '--hdf_train', help="path to hdf for vocoder training with hdf instead of audio files")
    parser.add_argument('-v', '--hdf_val', help="path to hdf for vocoder training with hdf instead of audio files")
    parser.add_argument('-m', '--mel_basis', help="path to mel_basis for extraction with torch")

    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    pt_dir = os.path.join(hp.log.chkpt_dir, args.name)
    log_dir = os.path.join(hp.log.log_dir, args.name)
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (args.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    writer = MyWriter(hp, log_dir)

    # create temporary directory and copy audio files
    tmpdir = tempfile.TemporaryDirectory()
    shutil.copytree(hp.data.input_files, tmpdir.name, dirs_exist_ok=True)
    print("copied files to:", tmpdir.name)
    training_filelist, validation_filelist = get_dataset_filelist(hp, tmpdir.name)

    # initialize hdf sequences if available
    hdf_seq_t = hdf_tag_t = hdf_seq_val = hdf_tag_val = None
    if args.hdf_train and args.hdf_val:
        hdf_seq_t, hdf_tag_t = load_normal_data(args.hdf_train)
        hdf_seq_val, hdf_tag_val = load_normal_data(args.hdf_val)
    else:
        print("no hdf for training/validation available")
    trainloader = create_dataloader(hp, args, True, validation_filelist, training_filelist, hdf_seq_t, hdf_tag_t, hdf_seq_val, hdf_tag_val)
    valloader = create_dataloader(hp, args, False, validation_filelist, training_filelist, hdf_seq_t, hdf_tag_t, hdf_seq_val, hdf_tag_val)

    train(args, pt_dir, args.checkpoint_path, trainloader, valloader, writer, logger, hp, hp_str)
