import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import configparser
from tensorboardX import SummaryWriter

from cae.dataset import AstroDataset
from cae.prep import prep_loaders
from cae.epoch import (
    train_one_epoch, eval_one_epoch
)
from cae.utils import setup_config
from cae.models.AutoEncoder import Model


# -- Set up configuration including paths, hyperparameters, etc.
config = setup_config(sys.argv[1])

dpaths = config['dpaths']
dhypar = config['dhypar']
dpretr = config['dpretr']
dcommt = config['dcommt']

print(f'Training version: {dpaths["output_version"]}')


# -- Use gpu if it's available; otherwise use cpu
device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)
print(f'Using device: {device}')


# -- Set up output path to store trained model, predictions, etc.
model_output_path = os.path.join(
    dpaths['output_folder'],
    'fold-' + dhypar['test_id']
)
if not os.path.exists(model_output_path):
    os.makedirs(model_output_path)


# -- Generate data loaders
loader_maker = prep_loaders(config)
loaders = loader_maker.loaders


# -- Set up models with specified hyperparameters
# ---- model hyperparameters
hpar = {
    'lc_length':int(dhypar['mlp_in_length']),
    'latent_channel':int(dhypar['latent_size'])
}

# ---- optimizer hyperparameters
optim_hpar = {
    'lr':float(dhypar['lr']),
    'weight_decay':float(dhypar['wdecay'])
}
if dhypar['optim_type'] == 'sgd':
    optim_hpar['momentum'] = float(dhypar['moment'])

# ---- initialize CAE model
model = Model(
    model=dhypar['model_name'],
    optimizer=dhypar['optim_type'],
    scheduler=bool(dhypar['use_scheduler']),
    hpar=hpar,
    optimhpar=optim_hpar
)

# ---- encoder, decoder modules
encoder = model.encoder
decoder = model.decoder

# print('Encoder:')
# print(encoder)
# print('Decoder:')
# print(decoder)

# ---- move modules to gpu (if available)
encoder = encoder.to(device)
decoder = decoder.to(device)

# ---- optimizer and scheduler modules
optimizer = model.optimizer
scheduler = model.scheduler

# ---- loss function
if dhypar['loss_type'] == 'mae':
    criterion = nn.MAELoss()
elif dhypar['loss_type'] == 'mse':
    criterion = nn.MSELoss()
elif dhypar['loss_type'] == 'bce':
    criterion = nn.BCELoss()
elif dhypar['loss_type'] == 'poisson':
    criterion = nn.PoissonNLLLoss(log_input=dhypar['set_log'], eps=1e-5)
else:
    raise Exception(
        'The loss type is not yet implemented,'+\
        ' choose between mae, mse, bce'
    )

# ---- impose constrain or not
if dhypar['add_constrain'] == 'none':
    _add_constr = None
else:
    _add_constr = dhypar['add_constrain']

# ---- the scalar of the constrain term
if dhypar['add_weight'] == 'none':
    _add_weight = None
else:
    _add_weight = float(dhypar['add_weight'])


# -- Writer function to store all hypar-parameters,
#    loss functions, etc.
writer=SummaryWriter(
    os.path.join(
        dpaths['log_folder'], 'fold-'+dhypar['test_id']
    ),
    comment=dcommt['training_description']
)

loss_best_epoch = np.inf

metrics_best_epoch = {'best_epoch':0.,
                      'training_loss_best':0.,
                      'val_loss_best':0.,
                      'test_loss_best':0.}

writer.add_hparams(dhypar, metrics_best_epoch, name='tmp')


print('Start training')
for epoch in range(int(dhypar['epochs'])):
    training_loss, training_recon_loss, training_constr_loss = train_one_epoch(
        device, epoch, encoder, decoder, loaders['train_set'],
        optimizer, criterion, writer,
        num_constr_codes=int(dhypar['num_constr_codes']),
        add_constrain=_add_constr, loss_weight=_add_weight
    )
    _text = f'Epoch={epoch:<3d}  training loss: {training_loss:<10.3f}'
    _text += f' recon. loss: {training_recon_loss:<10.3f}'
    _text += f' const. loss: {training_constr_loss:<10.3f}'

    test_loss = eval_one_epoch(
        device, epoch, encoder, decoder, loaders['test_set'],
        criterion, writer=writer,
        output_dir=model_output_path,
        num_constr_codes=int(dhypar['num_constr_codes']),
        num_codes=int(dhypar['latent_size']),
        num_classes=dhypar['num_classes'],
        add_constrain=_add_constr, loss_weight=_add_weight
    )
    # for updating the scheduler and find the best epoch
    # if validation set is used, then this will be val loss
    _tmp_loss = test_loss
    
    if bool(dhypar['set_val']):
        val_loss = eval_one_epoch(
            device, epoch, encoder, decoder, loaders['val_set'],
            criterion, writer=writer,
            loss_tag='val epoch loss',
            output_dir=model_output_path,
            num_codes=int(dhypar['latent_size']),
            num_constr_codes=int(dhypar['num_constr_codes']),
            num_classes=dhypar['num_classes'],
            add_constrain=_add_constr, loss_weight=_add_weight
        )
        _tmp_loss = val_loss
        _text += ' val loss: {:<10.3f}'.format(val_loss)

    _text += ' test loss: {:<10.3f} learning rate: {:<10.5f}'.format(
        test_loss,optimizer.param_groups[0]['lr'])


    if _tmp_loss < loss_best_epoch:
        loss_best_epoch = _tmp_loss
        metrics_best_epoch['best_epoch']         = epoch
        metrics_best_epoch['training_loss_best'] = training_loss
        if bool(dhypar['set_val']):
            metrics_best_epoch['val_loss_best'] = val_loss
        metrics_best_epoch['test_loss_best'] = test_loss
    
    if dhypar['use_scheduler']:
        scheduler.step(_tmp_loss)
        
    torch.save(encoder.state_dict(), '%s/encoder_%d.pth' % (model_output_path, epoch))
    torch.save(decoder.state_dict(), '%s/decoder_%d.pth' % (model_output_path, epoch))

    print(_text)
    
writer.add_hparams(dhypar, metrics_best_epoch, name='hparams')
writer.close()