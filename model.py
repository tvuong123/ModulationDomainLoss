import torch.nn as nn
import torch


class GRUNet(nn.Module):
    """ RNN-based speech enhancement model used in [1]
        Estimates ratio mask for each T-F bin

        Parameters
        ----------
        input_dim: int
            Input dimension to the network
        output_dim: int
            Output dimension of the network
        hidden_dim: int
            RNN input and hidden unit dimension
        hidden_dim2: int
            Fully connected layer dimension
        num_layers: int
            Number of layers in the RNN
        bi: bool
            Bidirectional or unidirectional RNN
        dropout: int
            Dropout percentage in the RNN
        act_out: int
            Output mask activation


        [1] T. Vuong, Y. Xia, and R. M. Stern, “A modulation-domain lossfor neural-network-based real-time speech enhancement”
            Accepted ICASSP 2021, https://arxiv.org/abs/2102.07330



    """

    def __init__(self, input_dim=257, output_dim=257, hidden_dim=400, hidden_dim2=600, num_layers=2, bi=False, dropout=0.0, act_out=nn.Sigmoid()):

        super(GRUNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if bi:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True,
                              num_layers=num_layers, dropout=dropout, bidirectional=bi)
            mult = 2
        else:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True,
                              num_layers=num_layers, dropout=dropout, bidirectional=bi)
            mult = 1
        self.fc2 = nn.Linear(hidden_dim*mult, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc_out = nn.Linear(hidden_dim2, output_dim)
        self.act_out = act_out
        self.relu = nn.ReLU()

    def forward(self, x):
        """Calculate forward pass.
        Args:
            x (Tensor): Input STFT features (B, #frames, #input_dim).
        Returns:
            Tensor: STFT mask for enhancement (B, #frames, #output_dim)

            Typically input_dim == output_dim == #STFT freq_bin
        """
        x = self.relu(self.fc1(x))
        x, _ = self.gru(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc_out(x)
        return self.act_out(x)


if __name__ == '__main__':

    # Input data (B, T, F)
    noisy_stftM = torch.rand((5, 100, 257))
    model = GRUNet()
    enhanced_stftM = model(noisy_stftM) * noisy_stftM
