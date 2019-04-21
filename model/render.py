import torch

class Render(torch.nn.Module):
    """ Neural network for rendering module of O2P2
    """

    def __init__(self):
        super(Render, self).__init__()