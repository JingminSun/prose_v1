import logging
from .logger import create_logger

import os
import re
import sys
import math
import time
import json
import pickle
import random
import getpass
import argparse
import subprocess

import errno
import signal
from functools import wraps, partial
import torch
import numpy as np
import torch.distributed as dist
from collections import OrderedDict
from omegaconf import OmegaConf

# from logging import getLogger
# logger = getLogger()

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

DUMP_PATH = f"checkpoint/{getpass.getuser()}/dumped"
CUDA = True


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def load_json(filename):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def zip_dic(lst):
    dico = {}
    for d in lst:
        for k in d:
            if k not in dico:
                dico[k] = []
            dico[k].append(d[k])
    for k in dico:
        if isinstance(dico[k][0], dict):
            dico[k] = zip_dic(dico[k])
    return dico


def unsqueeze_dic(dico):
    dico_copy = {}
    for d in dico:
        if isinstance(dico[d], dict):
            dico_copy[d] = unsqueeze_dic(dico[d])
        else:
            dico_copy[d] = [dico[d]]
    return dico_copy


def squeeze_dic(dico):
    dico_copy = {}
    for d in dico:
        if isinstance(dico[d], dict):
            dico_copy[d] = squeeze_dic(dico[d])
        else:
            dico_copy[d] = dico[d][0]
    return dico_copy


def initialize_exp(params, write_dump_path=True):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    if write_dump_path:
        get_dump_path(params)
        if not os.path.exists(params.dump_path):
            os.makedirs(params.dump_path)

    # pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))
    OmegaConf.save(params, os.path.join(params.dump_path, "configs.yaml"))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith("--"):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match("^[a-zA-Z0-9_]+$", x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = " ".join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # prepare random seed
    if params.base_seed < 0:
        params.base_seed = np.random.randint(0, 1000000000)
    if params.test_seed < 0:
        params.test_seed = np.random.randint(0, 1000000000)

    # create a logger
    logger = create_logger(
        os.path.join(params.dump_path, "train.log"),
        rank=getattr(params, "global_rank", 0),
    )
    logger.info("============ Initialized logger ============")
    # logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info(OmegaConf.to_yaml(params, sort_keys=True))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    if not params.dump_path:
        params.dump_path = DUMP_PATH

    # create the sweep path if it does not exist
    sweep_path = os.path.join(params.dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    if not params.exp_id:
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(10))
            if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                break

        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()


# def to_cuda(item):
#     """ Recursively move tensor(s) to the CUDA device. """
#     if isinstance(item, list) or isinstance(item, tuple):
#         # Recurse into list/tuple elements
#         return type(item)(to_cuda(subitem) for subitem in item)
#     else:
#         # Process single tensor
#         return item.cuda() if item is not None else None


def to_cuda(item,use_cpu=False):
    """ Recursively move tensor(s) or models to the CUDA device, if available. """
    if not CUDA or use_cpu:
        return item
    if isinstance(item, torch.nn.Module):
        return item.cuda()  # Move a model to CUDA
    elif isinstance(item, (list, tuple)):
            # Recurse into list/tuple elements
        return type(item)(to_cuda(subitem) for subitem in item)
    elif isinstance(item, torch.Tensor):
        return item.cuda()  # Move a tensor to CUDA
    else:
        raise TypeError("Item type cannot be moved to CUDA: {}".format(type(item)))

# def sync_dict(d):
#     """
#     Synchronize dictionary values across processes, d should be an order dict
#     """
#     res = OrderedDict()
#     lst_sync = torch.Tensor([v for _, v in d.items()]).cuda()

#     dist.barrier()
#     dist.all_reduce(lst_sync, op=dist.ReduceOp.SUM)

#     idx = 0
#     for k in d.keys():
#         res[k] = lst_sync[idx].item()
#         idx += 1

#     return res


def sync_tensor(t):
    """
    Synchronize a tensor across processes
    """
    device = t.device
    t_sync = t.cuda()

    dist.barrier()
    dist.all_reduce(t_sync, op=dist.ReduceOp.SUM)

    return t_sync.to(device)


class MyTimeoutError(BaseException):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(repeat_id, signum, frame):
            # logger.warning(f"Catched the signal ({repeat_id}) Setting signal handler {repeat_id + 1}")
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise MyTimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator


# if __name__ == '__main__':
#     a = torch.from_numpy(np.random.randn(1,10))
#
#     b = [a, [a for _ in range(2)]]
#
#     c = to_cuda(b)
#
#     print()