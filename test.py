import numpy as np
import scipy as sp
import argparse
import os
import sys

import torch

from cae.utils import read_csv
from cae.dataset import AstroDataset
from cae.models.AutoEncoder import Model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



parser = argparse.ArgumentParser(
    description="Extracting/Reconstructing input NuSTAR spectra using pretrained CAE"
)
parser.add_argument(
    '--label',
    type=str, required=True,
    help="Label file containing all required info to load spectrum (required)"
)
parser.add_argument(
    '--output',
    type=str, default=os.path.join(os.getcwd(), 'results.npz'),
    help="Output file containing input spec, reconstructed spec, " +\
         "input label, and predicted label (default: ./results.npz)"
)
parser.add_argument(
    '--batch',
    type=int, default=10,
    help="Number of spectra per batch (default: 10)"
)



args = parser.parse_args()


hpar = {
    'num_layers':4,
    'latent_channel':4,
    'lc_length':2960
}
num_classes = hpar['latent_channel']
num_codes = hpar['latent_channel']

model = Model(
    model='cae',
    optimizer='adam',
    scheduler=False,
    hpar=hpar,
    run='predict'
)

pre_enc = model.encoder
pre_dec = model.decoder

pretrain_encoder_path = os.path.join(
    os.getcwd(), 'pretrained', f'encoder_278.pth'
)
pretrain_decoder_path = os.path.join(
    os.getcwd(), 'pretrained', f'decoder_278.pth'
)

pre_enc.load_state_dict(torch.load(pretrain_encoder_path))
pre_enc = pre_enc.to(device)

pre_dec.load_state_dict(torch.load(pretrain_decoder_path))
pre_dec = pre_dec.to(device)


testset = AstroDataset(
    label_file=args.label,
    mask_en=[0., 120.],
    class_idx=[1,2,3,4],
    setlog=False,
    epsilon=1e-5,
    norm_method={'method':'const', 'norm_const':1.}
)

loader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batch,
    shuffle=False, num_workers=1,
    drop_last=False
)



pre_enc.eval()
pre_dec.eval()


test_hat = []
for i in range(num_codes):
    test_hat.append([])

test_hat_out = []

test_true = []
for i in range(num_classes):
    test_true.append([])

test_true_out = []

with torch.no_grad():
    for step, (x, y) in enumerate(loader):
        b_x = x.clone().float().to(device)
        b_y = y.clone().float().to(device)

        b_x = torch.unsqueeze(b_x, 1)

        code = pre_enc(b_x).squeeze()
        output = pre_dec(code)

        b_x = torch.squeeze(b_x, 1)
        output = torch.squeeze(output, 1)

        if step == 0:
            test_recon = output
            test_input = b_x
        else:
            test_recon = torch.cat((test_recon, output), 0)
            test_input = torch.cat((test_input, b_x), 0)

        for i in range(num_codes):
            if step == 0:
                test_hat[i] = code[:,i]
            else:
                test_hat[i] = torch.cat((test_hat[i], code[:,i]), 0)

        for i in range(num_classes):
            if step == 0:
                test_true[i] = b_y[:,i]
            else:
                test_true[i] = torch.cat((test_true[i], b_y[:,i]), 0)


test_recon_out = test_recon.detach().cpu().numpy()
test_input_out = test_input.detach().cpu().numpy()

for i in range(num_codes):
    test_hat_out.append(test_hat[i].detach().cpu().numpy())

for i in range(num_classes):
    test_true_out.append(test_true[i].detach().cpu().numpy())
    
test_hat_out = np.array(test_hat_out)
test_true_out = np.array(test_true_out)

np.savez(
    args.output,
    spec=test_input_out, recon=test_recon_out,
    label=test_true_out, code=test_hat_out
)