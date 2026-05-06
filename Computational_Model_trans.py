import numpy as np
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
from importlib import reload
reload(nn)


class MSIdataset(Dataset):
    def __init__(self, data, xLoc, yLoc, zLoc):
        self.len = len(data)
        self.data = data
        self.xLoc = xLoc
        self.yLoc = yLoc
        self.zLoc = zLoc

    def __getitem__(self, index):
        return self.data[index], self.xLoc[index], self.yLoc[index], self.zLoc[index]

    def __len__(self):
        return self.len


class Model_trans(nn.Module):
    def __init__(self, d_mz, d_model=256, encoder_layer_num=7, use_decoder=False,
                 decoder_layer_num=7, n_head=8, device='cpu',
                 frac_alpha=0.97, noise_lambda=0.01, noise_alpha=6,
                 latent_gamma=0.001):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_mz, 512, device=device),
            nn.BatchNorm1d(num_features=512, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Linear(512, d_model, device=device),
            nn.BatchNorm1d(num_features=d_model, momentum=0.99, eps=1e-3))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head)
        self.transformerencoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=encoder_layer_num)
        self.recover = nn.Sequential(
            nn.Linear(d_model, 512), nn.BatchNorm1d(num_features=512, momentum=0.99, eps=1e-3),
            nn.ReLU(), nn.Linear(512, d_mz), nn.Softmax(dim=-1))
        self.d_model = d_model
        self.position_Enbedding = None
        self.log = {'training loss': [],
                    'testing loss': []}

        # ===== 分数阶非局部记忆机制 (FracRNN) =====
        self.frac_alpha = frac_alpha
        self.frac_A = nn.Parameter(torch.ones(1, device=device) * 0.1)
        self.frac_B = nn.Parameter(torch.zeros(1, device=device))
        self.frac_theta = nn.Parameter(torch.ones(1, device=device))
        self.register_buffer('frac_weights', self._build_frac_weights(d_model, device))

        # ===== 受控噪声注入参数 =====
        self.noise_lambda = noise_lambda
        self.noise_alpha = noise_alpha
        self.latent_gamma = latent_gamma

    def _build_frac_weights(self, L, device):
        idx = torch.arange(L, device=device).float()
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        W = torch.where(diff >= 0, (diff + 1) ** (self.frac_alpha - 1),
                        torch.tensor(0.0, device=device))
        W = W / W.sum(dim=1, keepdim=True).clamp(min=1e-10)
        return W

    def _frac_rnn_forward(self, z):
        G_z = self.frac_A * torch.tanh(self.frac_theta * z) + self.frac_B
        u = G_z @ self.frac_weights.T
        return u

    def _input_noise(self, x):
        std_x = x.std(dim=-1, keepdim=True).detach()
        eta = self.noise_lambda * torch.exp(-self.noise_alpha * std_x)
        return x + torch.randn_like(x) * eta

    def _latent_noise(self, h):
        return h + self.latent_gamma * torch.randn_like(h)

    def encode(self, input):
        x = self._input_noise(input) if self.training else input
        h = self.backbone(x)
        h = self._latent_noise(h) if self.training else h
        z = self.transformerencoder(h)
        u = self._frac_rnn_forward(z)
        h = z + u
        h = self._latent_noise(h) if self.training else h
        return h

    def forward(self, input):
        h = self.encode(input)
        return self.TIC_norm(self.recover(h))

    def TIC_norm(self, input):
        return input / torch.sum(input, dim=-1)[:, None]


def categorical_crossentropy(pred, label):
    pred = torch.clip(pred, min=1e-7, max=1. - 1e-7)
    loss = (torch.sum(-label * torch.log(pred))) / pred.shape[0]
    return loss
