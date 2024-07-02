import os
import csv
import numpy as np
import torch
import torch.nn as nn
import configparser
from configparser import ExtendedInterpolation
from torch.utils.data import Dataset
from .losses import loss_w_constrain

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def setup_config(fn, save=True):
    config = configparser.ConfigParser(
        interpolation=ExtendedInterpolation()
    )
    config.read(fn)

    if save:
        outversion = config['data']['output_version']
        foldindex  = config['hypar']['test_id']
        
        outfilename = os.path.join(
            os.getcwd(), 'configs_train', f'{outversion}-train-{foldindex}.ini'
        )
        with open(outfilename, 'w') as outf:
            config.write(outf)

    # convert the configs to dicts
    dpaths = dict(config['data'])
    dhypar = dict(config['hypar'])
    dpretr = dict(config['pretrain'])
    dcommt = dict(config['comment'])

    # the loaded items are strings, convert
    # them into float
    erng = dhypar['energy_range']
    erng = erng.split(',')
    erng = [float(x) for x in erng]
    dhypar['energy_range'] = erng
    dhypar['energy_range_high'] = erng[1]

    # convert comma-separate string into array
    # below is the class tag
    ctag = dhypar['class_tag']
    l_ctag = ctag.split(',')
    l_ctag = [int(x) for x in l_ctag]
    dhypar['class_tag'] = l_ctag
    dhypar['class_tags_str'] = ctag
    dhypar['num_classes'] = len(l_ctag)
    
    # global minimum and maximum
    minmax = dhypar['norm_min_max']
    l_minmax = minmax.split(',')
    l_minmax = [float(x) for x in l_minmax]
    dhypar['norm_min_max'] = l_minmax
    dhypar['norm_min_max_str'] = minmax
    
    # boolean has to be loaded specifically
    # using getboolean command
    booleans = ['cross_valid',
                'set_val',
                'pretrain', 
                'set_log', 
                'use_scheduler']
    for key in booleans:
        k = config['hypar'].getboolean(key)
        dhypar[key] = k
        
    # for pretrained weights, write into two different
    # paths for encoder and decoder correspondingly
    load_pretr = config['pretrain'].getboolean('load_pretrain')
    dpretr['load_pretrain'] = load_pretr
    dpretr['enc_pretr'] = os.path.join(
        dpretr['pre_folder'],
        'fold-{}'.format(dpretr['fold']),
        'encoder_{}.pth'.format(dpretr['model'])
    )
    dpretr['dec_pretr'] = os.path.join(
        dpretr['pre_folder'],
        'fold-{}'.format(dpretr['fold']),
        'decoder_{}.pth'.format(dpretr['model'])
    )
    
    # parent dict
    dconfig = {'dpaths':dpaths,
               'dhypar':dhypar,
               'dpretr':dpretr,
               'dcommt':dcommt}
    
    return dconfig

def gen_class_tag(list_classes):
    loss_prefix = "test loss "
    epoch_loss_prefix = "test epoch loss "
    abs_loss_prefix = "mean abs loss "

    loss_tag, epoch_loss_tag, abs_loss_tag = [], [], []

    for i in range(len(list_classes)):
        loss_tag.append(loss_prefix+list_classes[i]) 
        epoch_loss_tag.append(epoch_loss_prefix+list_classes[i]) 
        abs_loss_tag.append(abs_loss_prefix+list_classes[i]) 

    return loss_tag, epoch_loss_tag, abs_loss_tag

def read_csv(path):
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = np.asarray(data)
    return data

def _get_first_layer(model):
    # check the first layer and see if we need to
    # unsqueeze the input
    _first_layer = list(dict(model.named_children()).values())[0]
    
    if isinstance(_first_layer, nn.Linear):
        return 'linear'
    
    elif isinstance(_first_layer, nn.Sequential):
        
        if isinstance(_first_layer[0], nn.Linear):
            return 'linear'
        else:
            return 'conv'
        
    else:
        return 'conv'