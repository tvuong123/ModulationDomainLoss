import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import math
from pdb import set_trace


CWD = os.path.dirname(__file__)

class GRUEnhancer(nn.Module):
    def __init__(self, device='cpu'):

        super(GRUEnhancer, self).__init__()        
        self.device = device
        self.hparams = np.load(os.path.join(CWD, 'hparams.npy'), allow_pickle=True)[()]
        self.model = GRUNet(numlayers=2,bi=False, dropout=0).to(self.device)
        self.norm_function = lambda x : mvnorm1(x, .01, tau=3, tau_init=.1, t_init=.1)
        self.nfft = int(self.hparams.stft['nfft'])
        self.window_len = int(self.hparams.stft['window_length'])
        self.hop_len = int(self.hparams.stft['hop_len'])
        self.norm = self.hparams.stft['norm']
        self.eps = 1e-6
        print(self.model)

    def __call__(self, signal):
        with torch.no_grad():
            torch_signal = torch.Tensor(signal).unsqueeze(0)
            wave = torch_signal.to(self.device)
            stft_noisy_mag, stft_noisy_phase = torchaudio.functional.magphase(torch.stft(wave, n_fft=self.nfft, hop_length=self.hop_len, win_length=self.window_len, window=torch.hamming_window(self.window_len).to(self.device)))
            stft_input_mag = torch.transpose(stft_noisy_mag.clone(), 2,1).cpu().numpy()

            stft_input_power = stft_input_mag[0]**2
            smallpower = stft_input_power < self.eps
            stft_input_power[smallpower] = np.log(self.eps)
            stft_input_power[~smallpower] = np.log(stft_input_power[~smallpower])
            stft_input_power = self.norm_function(stft_input_power)

            stft_input_power = torch.FloatTensor(stft_input_power).to(self.device)
            mask = self.model(stft_input_power.unsqueeze(0))
            enhanced_mag = torch.transpose(mask, 2, 1) * stft_noisy_mag
       
            complex_enhanced = torch.zeros((enhanced_mag.shape[0],enhanced_mag.shape[1], enhanced_mag.shape[2],2))
            complex_enhanced[:,:,:,0] = enhanced_mag * torch.cos(stft_noisy_phase)
            complex_enhanced[:,:,:,1] = enhanced_mag * torch.sin(stft_noisy_phase)

            enhanced_sig_recon = torch.istft(complex_enhanced, n_fft=self.nfft, hop_length=self.hop_len, win_length=self.window_len, window=torch.hamming_window(self.window_len))
            
        return enhanced_sig_recon


class GRUNet(nn.Module):

    def __init__(self, numlayers,bi, dropout, actout=nn.Sigmoid(), input_dim=257):
     
        super(GRUNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 400)
        if bi:
            self.gru = nn.GRU(400, 400, batch_first=True, num_layers=numlayers, dropout=dropout,bidirectional=bi)
            mult = 2
        else:
            self.gru = nn.GRU(400, 400, batch_first=True, num_layers=numlayers, dropout=dropout,bidirectional=bi)
            mult = 1
        self.fc2 = nn.Linear(400*mult, 600)
        self.fc3 = nn.Linear(600, input_dim)
        self.fcExtra = nn.Linear(600,600)
        self.actout = actout
        self.relu = nn.ReLU()
                
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x, _ = self.gru(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.relu(self.fcExtra(x))
        x = self.fc3(x)
        return self.actout(x)
        
def mvnorm1(powspec, frameshift, tau=3., tau_init=.1, t_init=.2):
    """Online mean and variance normalization of a short-time power spectra.

    This function computes online mean/variance as a scalar instead of a vector
    in `mvnorm`.

    Parameters
    ----------
    powspec: numpy.ndarray
        Real-valued short-time power spectra with dimension (T,F).
    frameshift: float
        Number of seconds between adjacent frame centers.

    Keyword Parameters
    ------------------
    tau: float, 3.
        Time constant of the median-time recursive averaging function.
    tau_init: float, .1
        Initial time constant for fast adaptation.
    t_init: float, .2
        Amount of time in seconds from the beginning during which `tau_init`
        is applied.
        The rest of time will use `tau`.

    Returns
    -------
    powspec_norm: numpy.ndarray
        Normalized short-time power spectra with dimension (T,F).

    """
    alpha = np.exp(-frameshift / tau)
    alpha0 = np.exp(-frameshift / tau_init)  # fast adaptation
    init_frames = math.ceil(t_init / frameshift)
    assert init_frames < len(powspec)

    mu = np.empty(len(powspec))
    var = np.empty(len(powspec))
    # Start with global mean and variance
    mu[0] = alpha0 * powspec.mean() + (1-alpha0)*powspec[0].mean()
    var[0] = alpha0 * (powspec**2).mean() + (1-alpha0)*(powspec[0]**2).mean()
    for ii in range(1, init_frames):
        mu[ii] = alpha0*mu[ii-1] + (1-alpha0)*powspec[ii].mean()
        var[ii] = alpha0*var[ii-1] + (1-alpha0)*(powspec[ii]**2).mean()
    for ii in range(init_frames, len(powspec)):
        mu[ii] = alpha*mu[ii-1] + (1-alpha)*powspec[ii].mean()
        var[ii] = alpha*var[ii-1] + (1-alpha)*(powspec[ii]**2).mean()

    return (powspec - mu[:, np.newaxis]) / np.maximum(
        np.sqrt(np.maximum(var[:, np.newaxis]-mu[:, np.newaxis]**2, 0)), 1e-12)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--in_file_name', type=str, default='test.wav',   help='full input file path')
    parser.add_argument('--out_file_name', type=str, default='enhanced.wav', help='full output file path')
    parser.add_argument('--use_gpu', type=int, default=0, help='1 if usinig gpu, 0 if cpu')
    parser.add_argument('--norm', type=int, default=0, help='1 will normalize waveform to -1 to 1')

    args = parser.parse_args()
    in_file_name = args.in_file_name
    out_file_name = args.out_file_name
    gpu = bool(int(args.use_gpu))
    norm = bool(int(args.norm))
    if gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    loaded_weights = torch.load('se_model_weights.ckpt', map_location=device)
    enhancer = GRUEnhancer(device)
    enhancer.load_state_dict(loaded_weights['state_dict'],strict=False)
    audio, sr = sf.read(in_file_name)
    assert sr==16000,'can only enhance 16k sampling rate'
    print('Enhancing [{}] ==> [{}]'.format(in_file_name, out_file_name))

    if norm:
        audio /= np.abs(audio).max()

    enhanced = enhancer(audio).squeeze(0).cpu().numpy()

    sf.write(out_file_name, enhanced, sr)
 





