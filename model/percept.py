import torch

class Percept(torch.nn.Module):
    """ Neural network for perception module of O2P2
    """

    def __init__(self):
        super(Percept, self).__init__()