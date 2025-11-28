import torch
import torch.distributed as dist
from torch.distributions.uniform import Uniform

import os
import re
import sys
import math
import logging
from copy import deepcopy
from collections import OrderedDict
import random
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def randint(low, high):
    return int(torch.randint(low, high, (1, )))


def rand_uniform(low, high):
    return float(Uniform(low, high).sample())


def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


def update_lr_cos(nb_iter, warm_up_iter, total_iter, max_lr, optimizer, min_lr=1e-7):

    if nb_iter < warm_up_iter:
        current_lr = max_lr * (nb_iter + 1) / (warm_up_iter + 1)
    else:
        current_lr = min_lr + (max_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * nb_iter / (total_iter - warm_up_iter)))

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


class CTCLabelConverter(object):
    def __init__(self, character):
        dict_character = list(character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1
        if len(self.dict) == 87:     # '[' and ']' are not in the test set but in the training and validation sets.
            self.dict['['], self.dict[']'] = 88, 89
        self.character = ['[blank]'] + dict_character

    def encode(self, text):
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text).to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        texts = []
        index = 0

        for l in length:
            t = text_index[index:index + l]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])) and t[i]<len(self.character):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class Averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


class Metric(object):
    def __init__(self, name=''):
        self.name = name
        self.sum = torch.tensor(0.).double()
        self.n = torch.tensor(0.)

    def update(self, val):
        rt = val.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        self.sum += rt.detach().cpu().double()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n.double()


class ModelEma:
    def __init__(self, model, decay=0.9999, device='', resume=''):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path, mapl=None):
        checkpoint = torch.load(checkpoint_path,map_location=mapl)
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)
            print("=> Loaded state_dict_ema")
        else:
            print("=> Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model, num_updates=-1):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        if num_updates >= 0:
            _cdecay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        else:
            _cdecay = self.decay

        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * _cdecay + (1. - _cdecay) * model_v)


def format_string_for_wer(str):
    str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)
    str = re.sub('([ \n])+', " ", str).strip()
    return str

def load_checkpoint(model, model_ema, optimizer, checkpoint_path, logger):
    best_cer, best_wer, start_iter = 1e+6, 1e+6, 1
    train_loss, train_loss_count = 0.0, 0
    optimizer_state = None
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)

        # Load model state dict (handle module prefix like in test.py)
        model_dict = OrderedDict()
        pattern = re.compile('module.')

        # For main model, load from the 'model' state dict
        # (the training checkpoint contains both 'model' and 'state_dict_ema')
        if 'model' in checkpoint:
            source_dict = checkpoint['model']
            logger.info("Loading main model from 'model' state dict")
        elif 'state_dict_ema' in checkpoint:
            source_dict = checkpoint['state_dict_ema']
            logger.info(
                "Loading main model from 'state_dict_ema' (fallback)")
        else:
            raise KeyError(
                "Neither 'model' nor 'state_dict_ema' found in checkpoint")

        for k, v in source_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict[k] = v

        model.load_state_dict(model_dict, strict=True)
        logger.info("Successfully loaded main model state dict")

        # Load EMA state dict if available
        if 'state_dict_ema' in checkpoint and model_ema is not None:
            ema_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                if re.search("module", k):
                    ema_dict[re.sub(pattern, '', k)] = v
                else:
                    ema_dict[k] = v
            model_ema.ema.load_state_dict(ema_dict, strict=True)
            logger.info("Successfully loaded EMA model state dict")

        # Load optimizer state - handle SAM optimizer structure
        if 'optimizer' in checkpoint and optimizer is not None:
            try:
                optimizer_state = checkpoint['optimizer']
                logger.info(
                    "Optimizer state will be loaded after optimizer initialization")
            except Exception as e:
                logger.warning(f"Failed to prepare optimizer state: {e}")
                optimizer_state = None

        # Load metrics from checkpoint if available
        if 'best_cer' in checkpoint:
            best_cer = checkpoint['best_cer']
        if 'best_wer' in checkpoint:
            best_wer = checkpoint['best_wer']
        if 'nb_iter' in checkpoint:
            start_iter = checkpoint['nb_iter'] + 1

        # Parse CER, WER, iter from filename as fallback
        m = re.search(
            r'checkpoint_(?P<cer>[\d\.]+)_(?P<wer>[\d\.]+)_(?P<iter>\d+)\.pth', checkpoint_path)
        if m and 'best_cer' not in checkpoint:
            best_cer = float(m.group('cer'))
            best_wer = float(m.group('wer'))
            start_iter = int(m.group('iter')) + 1

        if 'train_loss' in checkpoint:
            train_loss = checkpoint['train_loss']
        if 'train_loss_count' in checkpoint:
            train_loss_count = checkpoint['train_loss_count']
        if 'random_state' in checkpoint:
            random.setstate(checkpoint['random_state'])
            logger.info("Restored random state")
        if 'numpy_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_state'])
            logger.info("Restored numpy random state")
        if 'torch_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_state'])
            logger.info("Restored torch random state")
        if 'torch_cuda_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['torch_cuda_state'])
            logger.info("Restored torch cuda random state")

        # Validate that the model was loaded correctly by checking a few parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {total_params} total parameters")

        logger.info(
            f"Resumed best_cer={best_cer}, best_wer={best_wer}, start_iter={start_iter}")
    return best_cer, best_wer, start_iter, optimizer_state, train_loss, train_loss_count