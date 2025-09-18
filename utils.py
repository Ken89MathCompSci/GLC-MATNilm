import logging
import logging.handlers
import os
from datetime import datetime
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Sampler
import torch
import pickle as pkl

#LOGGER
def setup_log(subName='', tag='root'):
    # create logger
    logger = logging.getLogger(tag)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    # file log
    log_name = tag + datetime.now().strftime('log_%Y_%m_%d.log')

    log_path = os.path.join('log', subName, log_name)
    fh = logging.handlers.RotatingFileHandler(log_path, mode='a', maxBytes=100 * 1024 * 1024, backupCount=1, encoding='utf-8')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger

# DIRECTORIES
def mkdir(dirName):
    if not os.path.exists(dirName):
        os.system('mkdir {}'.format(dirName.replace('/', '\\')))

def mkdirectory(subName, saveModel):
    dirName_data = "data/" + subName
    dirName_log = "log/" + subName
    mkdir(dirName_data)
    mkdir(dirName_log)

    if saveModel is True:
        model_name = "/s0"
        dirName_model = "history_model/" + subName + model_name
        mkdir(dirName_model)
        return dirName_model

# DATALOADER
def data_loader(args):
    path = "data/redd/"
    train_arrays = pkl.load(open(os.path.join(path,"train_small.pkl"),'rb'))
    val_arrays = pkl.load(open(os.path.join(path,"val_small.pkl"),'rb'))
    test_arrays = pkl.load(open(os.path.join(path,"test_small.pkl"),'rb'))
    test_arrays = test_arrays[4:5]

    length_input = args.inputLength
    length_output = args.outputLength

    if args.debug:
        train_arrays[0] = train_arrays[0][:2000]

    ListTrain = [SubSet(array.values, length_input=length_input, length_output=length_output) for array in train_arrays]
    ListVal = [SubSet(array.values, length_input=length_input, length_output=length_output) for array in val_arrays]
    ListTest = [SubSet(array.values, length_input=length_input, length_output=length_output) for array in test_arrays]
    return ConcatDataset(ListTrain), ConcatDataset(ListVal), ConcatDataset(ListTest)

class SubSet(Dataset):
    def __init__(self, x, length_input=400, length_output=400, stride=1):
        super(SubSet, self).__init__()
        self.outputs = x[:, 1:]
        self.mains = x[:, :1]
        self.inLen = length_input
        self.outLen = length_output
        self.stride = stride

    def __getitem__(self, index):
        in_begin = index
        in_end = in_begin + self.inLen

        X = self.mains[in_begin:in_end,:]
        Y = self.outputs[in_begin:in_end,:]

        X_scaled = X / 612
        Y_scaled = Y / 612
        # Appliance-specific thresholds for better classification
        thresholds = [50, 10, 100, 100]  # dishwasher, fridge, microwave, washer
        Y_of = np.zeros_like(Y)
        for i in range(Y.shape[1]):
            threshold = thresholds[i] if i < len(thresholds) else 15
            Y_of[:, i] = np.where(Y[:, i] > threshold, 1, 0)
        return X, Y, X_scaled, Y_scaled, Y_of

    def __len__(self):
        return len(self.outputs) - self.inLen + 1
    
class testSampler(Sampler):
    def __init__(self, length, step):
        self.length = length
        self.step = step

    def __iter__(self):
        return iter(range(0,self.length,self.step))

    def __len__(self) -> int:
        return len(range(0,self.length,self.step))

# MODEL UTILS    
def saveModel(logger, net, path):
    torch.save({
        'model_state_dict': net.model.state_dict(),
        'model_optimizer_state_dict': net.model_opt.state_dict(),
    }, path)
    logger.info(f'Model saved')

def loadModel(logger, net, checkpoint):
    net.model.load_state_dict(checkpoint['model_state_dict'])
    net.model_opt.load_state_dict(checkpoint['model_optimizer_state_dict'])
    logger.info(f'Model loaded')
    return net

class EarlyStopping:
    def __init__(self, logger, patience=7, verbose=False, delta=0, best_score=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            self.early_stop = False

    def save_checkpoint(self, val_loss, net, path):
        if self.verbose:
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        saveModel(self.logger, net, path)
        self.val_loss_min = val_loss
