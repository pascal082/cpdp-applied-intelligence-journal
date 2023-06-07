import argparse
import tensorflow as tf
import time
from contextlib import redirect_stdout
from Config.defaults import *
from Domain.SoftwareDefect.softwaredefect import *
from Models.SoftwareDefect.run_model import *
from Domain.domain_type import TransferType, DomainType


'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
global cfg
global data


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi CNN ')

    parser.add_argument(
        '-config-file', '--config_file',
        '-data-csv', '--data_csv',
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        '-type', '--type',
        metavar="FILE",
        help="type of experiment",
        type=str,
    )
    args = parser.parse_args()
    return args

'''
function to set up result directory and process config file
'''
def setup(config_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    cfg.ROOT_DIR = ROOT_DIR + "/"

    # Example of using the cfg as global access to options
    # if cfg.SYSTEM.NUM_GPUS > 0:
    # my_project.setup_multi_gpu_support()

    # update the config file with model directory
    model_dir = os.path.join("Result", cfg.GENERAL.DOMAIN, cfg.GENERAL.DIRECTORY, cfg.DATA.INPUT)
    if cfg.TRAIN.TYPE == TransferType.Transfer:
        cfg.GENERAL.GLOBAL_MODEL_DIRECTORY = os.path.join("Result", cfg.GENERAL.DOMAIN, cfg.GENERAL.DIRECTORY,
                                                          cfg.DATA.INPUT)
        dirpath = os.path.join("Result", cfg.GENERAL.DOMAIN, cfg.GENERAL.DIRECTORY, cfg.DATA.INPUT, "transfer_learning")
    elif cfg.TRAIN.TYPE == TransferType.Global:
        dirpath = os.path.join("Result", cfg.GENERAL.DOMAIN, cfg.GENERAL.DIRECTORY, cfg.DATA.INPUT)

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('GPU device not found')
    else:
        print('Found GPU at: {}'.format(device_name))
    # with tf.device(device_name):

    cfg.GENERAL.MODEL_DIRECTORY = dirpath

    if not os.path.exists(cfg.GENERAL.MODEL_DIRECTORY):
        os.makedirs(cfg.GENERAL.MODEL_DIRECTORY)

    # save config file used
    with open(os.path.join(cfg.GENERAL.MODEL_DIRECTORY,
                           "{0}_{1}_args.json".format(cfg.GENERAL.DOMAIN, cfg.TRAIN.TYPE)),
              "w") as outfile:
        with redirect_stdout(outfile): print(cfg.dump())

    # freeze paramaters
    cfg.freeze()
    return cfg

'''
Train model
'''
def train(config_file):

    start_time = time.time()
    # 0. set random states
    np.random.seed(42)
    cfg=setup(config_file)

    # check domain and  get data
    if not cfg.GENERAL.USING_EXISTING_DATA_SPLIT:


        if cfg.GENERAL.DOMAIN == DomainType.NLP:
            print("Not added here")
        elif cfg.GENERAL.DOMAIN == DomainType.SOFWTAREDEFECT:
            data = SoftwareDefect(cfg)

    if cfg.GENERAL.DOMAIN == DomainType.SOFWTAREDEFECT:
        run_model(cfg, data)


    time_took = time.time() - start_time
    print(f"Total runtime: {hms_string(time_took)}")



if __name__ != 'main':
    args = parse_args()
    print(args.type)

    if args.type == "train":
        train(args.config_file)
    else:
        print("no type passed")



