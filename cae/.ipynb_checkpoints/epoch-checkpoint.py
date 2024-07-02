import os
import numpy as np
import torch
import torch.nn as nn
from .losses import loss_w_constrain
from .utils import _get_first_layer

def train_one_epoch(device, epoch, encoder, decoder, loader,
                    optimizer, criterion,
                    writer, write_summary=True,
                    loss_tag='training epoch loss',
                    num_constr_codes=4,
                    add_constrain=False, loss_weight=None):
    """
    Train one epoch by index, and return various losses.

    Parameters
    ----------
    - device: torch.device
        Managing to use CPU or GPU.
    - epoch: int
        Index of epoch.
    - encoder: pytorch.nn.Module
        Encoder model.
    - decoder: pytorch.nn.Module
        Decoder model.
    - loader: torch.utils.data.DataLoader
        Data loader.
    - optimizer: torch.optim.Optimizer
        Loss function optimizer
    - criterion: torch.nn
        Loss function
    - writer: tensorboardX.SummaryWriter
        Summary writer storing loss functions across
        each epoch
    - write_summary: boolean
        If true, store losses in the writer;
        otherwise, ignore
    - loss_tag: str
        the label (tag) stored together with the loss values
        indicating the loss whether it is training loss or test loss
    - num_constr_codes: int
        number of latent variables which are constrained
    - add_constrain: boolean
        include constrain term in the loss function
    - loss_weight: float
        weighing constraint term
        
    Returns
    -------
    - epoch_loss: torch.Tensor.float
        total loss (if including constrain) per epoch
    - epoch_reconstruct_loss: torch.Tensor.float
        reconstruction loss per epoch
    - epoch_constraint_loss: torch.Tensor.float
        constrain loss (if applied) per epoch
    """
    
    
    epoch_loss = 0
    epoch_reconstruct_loss = 0
    epoch_constraint_loss = 0

    encoder.train()
    decoder.train()

    firstLayerType = _get_first_layer(encoder)

    for step, (x, y) in enumerate(loader):

        b_x = x.clone().float().to(device)
        b_y = y.clone().float().to(device)

        if firstLayerType == 'conv':
            b_x = torch.unsqueeze(b_x, 1)
        
        encoder.zero_grad(set_to_none=True)
        decoder.zero_grad(set_to_none=True)


        code = encoder(b_x).squeeze()
        output = decoder(code)

        if firstLayerType == 'conv':
            b_x = torch.squeeze(b_x, 1)
            output = torch.squeeze(output, 1)

        loss, l1, l2 = loss_w_constrain(
            criterion,
            output, b_x,
            b_y, code,
            device,
            num_constr_codes,
            add_constrain, loss_weight
        )
        
        epoch_loss += loss.item()
        epoch_reconstruct_loss += l1.item()
        epoch_constraint_loss += l2.item()

        loss.backward()
        optimizer.step()

    epoch_loss /= (step+1)
    epoch_reconstruct_loss /= (step+1)
    epoch_constraint_loss /= (step+1)
    
    if write_summary:
        writer.add_scalar(loss_tag, epoch_loss, epoch)
        writer.add_scalar(loss_tag + ' recon', epoch_reconstruct_loss, epoch)
        writer.add_scalar(loss_tag + ' const', epoch_constraint_loss, epoch)

    return epoch_loss, epoch_reconstruct_loss, epoch_constraint_loss


def eval_one_epoch(device, epoch, encoder, decoder, loader,
                   criterion,
                   writer=None, write_summary=True,
                   loss_tag='test epoch loss',
                   num_codes=32, num_constr_codes=4,
                   add_constrain=False, loss_weight=False,
                   save_recon=True, save_code=True, save_true=True,
                   num_classes=4, output_dir=None):
    """
    Evaluate/test one epoch by index, and return various losses.

    Parameters
    ----------
    - device: torch.device
        Managing to use CPU or GPU.
    - epoch: int
        Index of epoch.
    - encoder: pytorch.nn.Module
        Encoder model.
    - decoder: pytorch.nn.Module
        Decoder model.
    - loader: torch.utils.data.DataLoader
        Data loader.
    - criterion: torch.nn
        Loss function.
    - writer: tensorboardX.SummaryWriter
        Summary writer storing loss functions across
        each epoch.
    - write_summary: boolean
        If true, store losses in the writer;
        otherwise, ignore.
    - loss_tag: str
        The label (tag) stored together with the loss values
        indicating the loss whether it is training loss or test loss.
    - num_constr_codes: int
        Number of latent variables which are constrained.
    - add_constrain: boolean
        Include constrain term in the loss function.
    - loss_weight: float
        Weighing constraint term.
    - save_recon: boolean
        Store reconstructed spectra from decoder in a .npy file.
    - save_code: boolean
        Store extracted latent code in a .npy file.
    - save_true: boolean
        Store normalized input label in a .npy file.
    - num_classes: int
        Number of input label.

    Returns
    -------
    - epoch_loss: torch.Tensor.float
        total loss (if including constrain) per epoch
    """

    epoch_loss = 0

    encoder.eval()
    decoder.eval()

    firstLayerType = _get_first_layer(encoder)
    
    # tag used as filename prefix
    _tag = loss_tag.split()[0]

    if save_code:
        test_hat = []
        for i in range(num_codes):
            test_hat.append([])

        test_hat_out = []

    if save_true:
        test_true = []
        for i in range(num_classes):
            test_true.append([])

        test_true_out = []

    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            b_x = x.clone().float().to(device)
            b_y = y.clone().float().to(device)

            if firstLayerType == 'conv':
                b_x = torch.unsqueeze(b_x, 1)
            
            code = encoder(b_x).squeeze()
            output = decoder(code)
            

            if firstLayerType == 'conv':
                b_x = torch.squeeze(b_x, 1)
                output = torch.squeeze(output, 1)

            loss, l1, l2 = loss_w_constrain(
                criterion,
                output, b_x,
                b_y, code,
                device,
                num_constr_codes,
                add_constrain, loss_weight
            )
            
            epoch_loss += loss.item()

            if save_recon:
                if step == 0:
                    test_recon = output
                else:
                    test_recon = torch.cat((test_recon, output), 0)

            if save_code:
                for i in range(num_codes):
                    if step == 0:
                        test_hat[i] = code[:,i]
                    else:
                        test_hat[i] = torch.cat((test_hat[i], code[:,i]), 0)
                        
            if save_true:
                for i in range(num_classes):
                    if step == 0:
                        test_true[i] = b_y[:,i]
                    else:
                        test_true[i] = torch.cat((test_true[i], b_y[:,i]), 0)

        epoch_loss /= (step+1)

    if write_summary:
        writer.add_scalar(loss_tag, epoch_loss, epoch)

    if save_recon:
        test_recon_out = test_recon.detach().cpu().numpy()
        np.save(
            os.path.join(output_dir, f'{_tag}_x_recon_{epoch}.npy'),
            np.array(test_recon_out)
        )

    if save_code:
        for i in range(num_codes):
            test_hat_out.append(test_hat[i].detach().cpu().numpy())

        np.save(
            os.path.join(output_dir, f'{_tag}_code_{epoch}.npy'),
            np.array(test_hat_out)
        )
    
    if save_true:
        for i in range(num_classes):
            test_true_out.append(test_true[i].detach().cpu().numpy())

        np.save(
            os.path.join(output_dir, f'{_tag}_y_true_{epoch}.npy'),
            np.array(test_true_out)
        )
    
    return epoch_loss