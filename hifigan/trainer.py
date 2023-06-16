import os
import time
import logging
import argparse
import tempfile
import shutil
from utils.train import train
from utils.hparams import HParam
from utils.writer import MyWriter
from datasets.dataloader import create_dataloader, get_dataset_filelist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
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

    # Needs to work with hop size of 0.05
    # assert hp.audio.hop_length == 256, \
    #    'hp.audio.hop_length must be equal to 256, got %d' % hp.audio.hop_length

    # create temporary directory and copy audio files
    tmpdir = tempfile.TemporaryDirectory()
    shutil.copytree(hp.data.input_files, tmpdir.name, dirs_exist_ok=True)
    print("copied files to:", tmpdir.name)
    training_filelist, validation_filelist = get_dataset_filelist(hp, tmpdir.name)

    trainloader = create_dataloader(hp, args, True, validation_filelist, training_filelist)
    valloader = create_dataloader(hp, args, False, validation_filelist, training_filelist)

    train(args, pt_dir, args.checkpoint_path, trainloader, valloader, writer, logger, hp, hp_str)
