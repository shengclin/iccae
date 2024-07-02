import os
import csv
import numpy as np
import torch

from .utils import read_csv
from torch.utils.data import Dataset


class AstroDataset(Dataset):
    """
    Dataset class for loading training/test spectra.
    Various of normalization options are available with
    specified hyperparameters.

    Parameters
    ----------
    - label_file: str
        Input label file.
    - mask_en: list
        Range of spectral energy to use.
    - class_idx: list
        Column indices of label file for loading specified.
        labels.
    - setLog: boolean
        If True, Set spectra in log-scale.
    - epsilon: float
        Small number added to input spectra to avoid negatives.
        Applied only if setLog is True.
    - norm_method: dict
        Python dictionary storing normalization/standardize method.
        Options are 
        - {'method':'const', 'norm_const':1e5}
        - {'method':'minmax'}
        - {'method':'globalminmax', 'min':, 'max':}
        - {'method':'standard', 'stand_mean':0.1, 'stand_std':0.01}
    """
    def __init__(self,
                 label_file=None,
                 mask_en=None,
                 class_idx=[1,2,3,4,5],
                 setlog=False, epsilon=1e-5,
                 norm_method={'method':'const','norm_const':1e5}):
        self.files = read_csv(label_file)
        self.classid = class_idx
        self.eps = epsilon
        self.setlog = setlog
        self.normethod = norm_method
        self.mask_en = mask_en

    def __getitem__(self, idx):
        count = self._load_spec(idx)
        
        if self.setlog:
            count = np.log10(count+self.eps)
            
        if self.normethod is not None:
            count = self._norm_spec(count)

        label = np.float32(self.files[idx,self.classid])

        return count, label

    def __len__(self):
        return len(self.files)
    
    def _norm_spec(self, count):
        if self.normethod['method'] == 'const':
            count /= self.normethod['norm_const']
        elif self.normethod['method'] == 'minmax':
            count = (count - count.min())/np.ptp(count)
        elif self.normethod['method'] == 'standard':
            count = (count - self.normethod['stand_mean']) / self.normethod['stand_std']
        elif self.normethod['method'] == 'globalminmax':
            count = count - self.normethod['norm_min_max'][0]
            count /= (self.normethod['norm_min_max'][1]-self.normethod['norm_min_max'][0])

        return count

    def _load_spec(self, idx):
        clsinfo   = self.files[idx]
        specfn    = clsinfo[0]
        spec      = np.load(specfn)
        en        = spec[:,0]
        cnt       = spec[:,2]

        if self.mask_en is not None:
            mask = (en>=self.mask_en[0])*(en<self.mask_en[1])
            mcnt = cnt[mask]

        return mcnt