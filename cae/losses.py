import torch
import torch.nn.functional as F


def loss_w_constrain(lossfn,
                     inputs, targets,
                     labels, codes,
                     device,
                     num_parts=4,
                     constrain=None, weight=None,):

    loss = lossfn(inputs, targets)
    N = labels.size(0)


    if constrain is not None:
        constr_range = constrain.split('_')[-1]
        if constr_range == 'part' and \
            codes.size(1) <= labels.size(1):
                raise ValueError('To put constraints on part of codes'+\
                                 ' make sure'+\
                                 ' the size of codes {}'.format(codes.size(1))+\
                                 ' is larger' +\
                                 ' than that of labels'+\
                                 ' {}'.format(labels.size(1)))

    if constrain == 'l1_all':
        regular = torch.sum(torch.abs(labels-codes))/N
    elif constrain == 'l1_part':
        regular = torch.sum(torch.abs(labels-codes[:,:num_parts]))/N
    elif constrain == 'l2_all':
        regular = torch.sum((labels-codes)**2)/N
    elif constrain == 'l2_part':
        regular = torch.sum((labels-codes[:,:num_parts])**2)/N
    else:
        regular = torch.tensor([0]).float().to(device)

    if weight is not None:
        regular *= weight

    return loss + regular, loss, regular
