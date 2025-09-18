import pickle as pkl
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.stats import truncnorm

class sigGen():
    def __init__(self, config):
        self.inLen = config.inputLength
        self.outLen = config.outputLength

        self.pool1 = pkl.load(open("data/redd/REDD_pool.pkl",'rb'))
        self.pool2 = pkl.load(open("data/redd/poolx.pkl",'rb'))
        self.pool = self.pool1 + self.pool2
        self.off = pkl.load(open("data/redd/offduration.pkl",'rb'))
        self.offduration = [[x for house in self.off for x in house[i]] for i in range(len(self.off[0]))]
        self.offduration = self.offduration + [[]] * len(self.pool2)

        self.offInt = [[x for x in list if x <= self.inLen] for list in self.offduration]

    def getMore(self, sigCh, y_of_ori, appNum):
        appSig = self.pool[appNum]

        houseNum2 = np.random.randint(len(appSig))
        samNum2 = np.random.randint(len(appSig[houseNum2]))
        sigCh2 = appSig[houseNum2][samNum2]
        y_of_ori2 = np.ones_like(sigCh2)
        if len(self.offInt[appNum]) > 100:
            zeroLength = self.offInt[appNum][np.random.randint(0,len(self.offInt[appNum]))]
        elif len(self.offInt[appNum]) > 10 and np.random.rand(1) < 0.7:
            zeroLength = self.offInt[appNum][np.random.randint(0,len(self.offInt[appNum]))]
        else:
            zeroLength = np.random.randint(0, self.length/4)

        zeroBetween = np.zeros(zeroLength)
        sigCh = np.concatenate((sigCh, zeroBetween, sigCh2), axis=None)
        y_of_ori = np.concatenate((y_of_ori, zeroBetween, y_of_ori2), axis=None)

        return sigCh, y_of_ori

    def getSignal(self, appNum, length):
        self.length = length
        appSig = self.pool[appNum]
        houseNum = np.random.randint(len(appSig))
        samNum = np.random.randint(len(appSig[houseNum]))
        sigCh = appSig[houseNum][samNum]
        y_of_ori = np.ones_like(sigCh)

        while len(sigCh) < self.outLen * 2:
            if np.random.rand(1) < 0.5:
                break
            sigCh, y_of_ori = self.getMore(sigCh, y_of_ori, appNum)

        signal = torch.from_numpy(sigCh)
        signal_scaled = signal/612
        y_of = torch.from_numpy(y_of_ori)
        return signal_scaled, signal, y_of

def vertScale(x):
    mu, sigma = 1, 0.2
    lower, upper = mu - 2*sigma, mu + 2*sigma
    tn = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    scale = tn.rvs(1)
    return x*torch.from_numpy(scale)

def horiScale(input):
    olength = int(len(input)/2)
    mu, sigma = 1, 0.2
    lower, upper = mu - 2*sigma, mu + 2*sigma
    tn = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    scale = tn.rvs(1)
    y = input.reshape(-1)
    x = np.arange(0, len(y))
    f = interp1d(x, y)

    xnew = np.arange(0, len(y) - 1, scale)
    ynew = f(xnew)
    if len(ynew) > olength:
        return torch.from_numpy(ynew[:olength])
    else:
        diffSize = olength - len(ynew)
        ynew = np.pad(ynew, (0, diffSize), 'constant')
        return torch.from_numpy(ynew)

def selectPortion(config, sig, length):
    insertIdxLrange = int((config.inputLength - config.outputLength)/2) - len(sig)
    insertIdxRrange = int((config.inputLength + config.outputLength)/2)
    idx = np.random.randint(insertIdxLrange, insertIdxRrange)
    # Insert signal
    signal = torch.zeros(length)
    if idx < 0:
        signal[:min(len(sig) + idx, length)] = sig[-idx:min(len(sig), length - idx)]
    else:
        signal[idx:min(len(sig) + idx, length)] = sig[:min(len(sig), length - idx)]
    return signal

def dataAug( X_scaled, Y_scaled, Y_of, sigClass, config):
    orilen = Y_scaled.shape[-1]
    prob=[config.prob0, config.prob1, config.prob2, config.prob3]
    xlen = len(sigClass.pool) - orilen
    prob.extend([0.1] * xlen)
    llen = Y_scaled.shape[1]

    # prob=[config.prob0, config.prob1, config.prob2]
    for i in range(X_scaled.shape[0]):
        minX = min(X_scaled[i,:,0]).item()
        for j in range(len(prob)):
            p = np.random.rand(1)
            if p < prob[j]:
                if j < orilen:
                    X_scaled[i,:,0] -= Y_scaled[i,:,j]
                    Y_scaled[i, :, j] -= Y_scaled[i,:,j]
                sig, sig_ori, y_of = sigClass.getSignal(j, Y_scaled.shape[1]*2)

                if j==0:
                    mode = np.random.choice(4, 1, p=[.25, .25, .25, .25])
                elif j==1:
                    mode = np.random.choice(4, 1, p=[.25, .25, .25, .25])
                else:
                    mode = np.random.choice(4, 1, p=[.25, .25, .25, .25])
                # 0: origin, 1:vertical scaled, 2: horizon scaled, 3: v&h scaled
                if mode == 1:
                    sig = vertScale(sig)
                elif mode == 2:
                    sig = horiScale(sig)
                elif mode == 3:
                    sig = horiScale(sig)
                    sig = vertScale(sig)

                sig = selectPortion(config, sig, llen)
                y_of = torch.where(sig > 15/612, torch.Tensor([1]), torch.Tensor([0]))
                if j < orilen:
                    Y_scaled[i,:,j] += sig
                    Y_of[i, :, j] = y_of
                X_scaled[i,:,0] += sig
    return X_scaled, Y_scaled, Y_of
