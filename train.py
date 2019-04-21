import torch
from torch.utils.data import DataLoader
from options.options import BaseOptions
from util.logger import Logger
import os.path as osp
from data.physNetReal import PhysNetReal
from data.dataset import O2P2Dataset

if __name__ == '__main__':
    opt = BaseOptions().parse()   # get options
    log_file = osp.join(opt.checkpoints_dir, "trainlog.txt")
    logger = Logger(log_file)
    use_gpu = torch.cuda.is_available()

    # Read and initialize dataset
    physNetData = PhysNetReal(opt.dataroot)

    # PyTorch Dataset classes for train, validation and test sets
    train_dataset = O2P2Dataset(physNetData.train, transform=None)
    val_dataset = O2P2Dataset(physNetData.val, transform=None)
    test_dataset = O2P2Dataset(physNetData.test, transform=None)

    # PyTorch Dataloaders for train, validation and test sets
    train_loader = DataLoader(train_dataset, batch_size=opt.train_batch_size, shuffle=True, pin_memory=use_gpu)
    val_loader = DataLoader(val_dataset, batch_size=opt.test_batch_size, shuffle=False, pin_memory=use_gpu)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, pin_memory=use_gpu)

