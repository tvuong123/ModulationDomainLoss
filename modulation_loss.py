import torch
import torch.nn as nn
import math
from torchaudio.transforms import MelScale
import torch.nn.functional as F

from pdb import set_trace


class ModulationDomainLossModule(torch.nn.Module):
    """Modulation-domain loss function developed in [1] for supervised speech enhancement

        In our paper, we used the gabor-based STRF kernels as the modulation kernels and used the log-mel spectrogram as the input spectrogram representation.
        Specific parameter details are in the paper and in the example below

        Parameters
        ----------
        modulation_kernels: nn.Module
            Differentiable module that transforms a spectrogram representation to the modulation domain

            modulation_domain = modulation_kernels(input_tf_representation)
            Input Spectrogram representation(B, T, F) --- (M) modulation_kernels---> Modulation Domain(B, M, T', F') 

        [1]

    """

    def __init__(self, modulation_kernels):
        super(ModulationDomainLossModule, self).__init__()

        self.modulation_kernels = modulation_kernels
        self.mse = nn.MSELoss(reduce=False)

    def forward(self, enhanced_spect, clean_spect):
        """Calculate modulation-domain loss
        Args:
            enhanced_spect (Tensor): spectrogram representation of enhanced signal (B, #frames, #freq_channels).
            clean_spect (Tensor): spectrogram representation of clean ground-truth signal (B, #frames, #freq_channels).
        Returns:
            Tensor: Modulation-domain loss value.
        """

        clean_mod = self.modulation_kernels(clean_spect)
        enhanced_mod = self.modulation_kernels(enhanced_spect)

        mod_mse_loss = self.mse(enhanced_mod, clean_mod)
        mod_mse_loss = torch.mean(torch.sum(mod_mse_loss, dim=(
            1, 2, 3))/torch.sum(clean_mod**2, dim=(1, 2, 3)))

        return mod_mse_loss


class GaborSTRFConv(nn.Module):
    """Gabor-STRF-based cross-correlation kernel."""

    def __init__(self, supn, supk, nkern, rates=None, scales=None):
        """Instantiate a Gabor-based STRF convolution layer.
        Parameters
        ----------
        supn: int
            Time support in number of frames. Also the window length.
        supk: int
            Frequency support in number of channels. Also the window length.
        nkern: int
            Number of kernels, each with a learnable rate and scale.
        rates: list of float, None
            Initial values for temporal modulation.
        scales: list of float, None
            Initial values for spectral modulation.
        """
        super(GaborSTRFConv, self).__init__()
        self.numN = supn
        self.numK = supk
        self.numKern = int(nkern * 2)

        if supk % 2 == 0:  # force odd number
            supk += 1
        self.supk = torch.arange(supk, dtype=torch.float32)
        if supn % 2 == 0:  # force odd number
            supn += 1
        self.supn = torch.arange(supn, dtype=self.supk.dtype)
        self.padding = (supn//2, supk//2)

        # Set up learnable parameters
        for param in (rates, scales):
            assert (not param) or len(param) == nkern
        if not rates:
            rates = torch.rand(nkern) * math.pi/2.0
        if not scales:
            scales = (torch.rand(nkern)*2.0-1.0) * math.pi/2.0

        self.rates_ = nn.Parameter(torch.Tensor(rates))
        self.scales_ = nn.Parameter(torch.Tensor(scales))

    def strfs(self):
        """Make STRFs using the current parameters."""
        if self.supn.device != self.rates_.device:  # for first run
            self.supn = self.supn.to(self.rates_.device)
            self.supk = self.supk.to(self.rates_.device)
        n0, k0 = self.padding
        nsin = torch.sin(torch.ger(self.rates_, self.supn-n0))
        ncos = torch.cos(torch.ger(self.rates_, self.supn-n0))
        ksin = torch.sin(torch.ger(self.scales_, self.supk-k0))
        kcos = torch.cos(torch.ger(self.scales_, self.supk-k0))
        nwind = .5 - .5 * \
            torch.cos(2*math.pi*(self.supn)/(len(self.supn)+1))
        kwind = .5 - .5 * \
            torch.cos(2*math.pi*(self.supk)/(len(self.supk)+1))
        strfr = torch.bmm((ncos*nwind).unsqueeze(-1),
                          (kcos*kwind).unsqueeze(1))

        strfi = torch.bmm((nsin*nwind).unsqueeze(-1),
                          (ksin*kwind).unsqueeze(1))

        return torch.cat((strfr, strfi), 0)

    def forward(self, sigspec):
        """Forward pass a batch of (real) spectra [Batch x Time x Frequency]."""
        if len(sigspec.shape) == 2:  # expand batch dimension if single eg
            sigspec = sigspec.unsqueeze(0)
        strfs = self.strfs().unsqueeze(1).type_as(sigspec)
        return F.conv2d(sigspec.unsqueeze(1), strfs, padding=self.padding)

    def __repr__(self):
        """Gabor filter"""
        report = """
            +++++ Gabor Filter Kernels [{}], supn[{}], supk[{}] +++++
        """.format(self.numKern, self.numN, self.numK
                   )
        return report


if __name__ == '__main__':

    # Example Usage

    gabor_strf_parameters = torch.load(
        'modulation_kernel_parameters/gabor_strf_parameters.pt', map_location=lambda storage, loc: storage)['state_dict']
    gabor_modulation_kernels = GaborSTRFConv(supn=30, supk=30, nkern=30)
    gabor_modulation_kernels.load_state_dict(gabor_strf_parameters)

    stft2mel = MelScale(n_mels=80, sample_rate=16000, n_stft=257)

    modulation_loss_module = ModulationDomainLossModule(
        gabor_modulation_kernels)

    # (B, F, T) - pytorch convention
    enhanced_speech_STFTM = torch.abs(torch.rand(5, 257, 100))
    clean_speech_STFTM = torch.abs(torch.rand(5, 257, 100))

    clean_log_mel = torch.log(torch.transpose(
        stft2mel(clean_speech_STFTM**2), 2, 1) + 1e-8)
    enhanced_log_mel = torch.log(torch.transpose(
        stft2mel(enhanced_speech_STFTM**2), 2, 1) + 1e-8)

    modulation_loss = modulation_loss_module(enhanced_log_mel, clean_log_mel)

    # modulation_loss.backward()
