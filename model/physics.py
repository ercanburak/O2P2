import torch

class Physics(torch.nn.Module):
    """ Neural network for physics module of O2P2
    """

    def __init__(self):
        super(Physics, self).__init__()