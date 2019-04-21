import torch
from torch.utils.data import DataLoader
from options.options import BaseOptions
from util.logger import Logger
import os.path as osp
from data.physNetReal import PhysNetReal
from data.dataset import O2P2Dataset
from model.percept import Percept
from model.physics import Physics
from model.render import Render
import time

def main():
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

    # Initialize model

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    percept = Percept()
    physics = Physics()
    render = Render()
    optimizer = torch.optim.Adam([{'params': percept.parameters()},
                                  {'params': physics.parameters()},
                                  {'params': render.parameters()}],
                                 lr=1e-3)

    # Start training
    for epoch in range(opt.max_epoch):
        start_time = time.time()

        # train for one epoch
        percept_loss, physics_loss, render_loss = train(train_loader, percept, physics, render, criterion, optimizer, use_gpu)

        elapsed_time = time.time() - start_time

        # print training details
        print_train_stats(logger, epoch, elapsed_time, percept_loss, physics_loss, render_loss)

    logger.log("Training completed.")

def print_train_stats(logger, epoch, elapsed_time, percept_loss, physics_loss, render_loss):
    """ Prints training details
    """
    logger.log('Epoch: [{0}]\t'
        'Time {epoch_time:.1f}\t'
        'Perception Loss {percept_loss:.4f}\t'
        'Physics Loss {physics_loss:.3f} \t'
        'Rendering Loss {render_loss:.4f}\t\t'.format(
        epoch+1, epoch_time=elapsed_time, percept_loss=percept_loss,
        physics_loss=physics_loss, render_loss=render_loss))

def train(train_loader, percept, physics, render, criterion, optimizer, use_gpu):
    """ Train the model for one epoch.
    """

    # switch to train mode
    percept.train()
    physics.train()
    render.train()

    for img0, img1, segs in train_loader:
        if use_gpu:
            img0, img1 = img0.cuda(), img1.cuda()
            for i, seg in enumerate(segs):
                segs[i] = seg.cuda()

        # TODO: compute model output

        # TODO: measure and record loss
        percept_loss = 0
        physics_loss = 0
        render_loss = 0
        # compute gradient and do optimizer step

        return percept_loss, physics_loss, render_loss

if __name__ == '__main__':
    main()