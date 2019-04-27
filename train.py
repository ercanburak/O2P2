import torch
from torch.utils.data import DataLoader
from torchvision.transforms import *
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
    torch.manual_seed(opt.seed);
    torch.cuda.manual_seed_all(opt.seed)

    # Read and initialize dataset
    phys_net_data = PhysNetReal(opt.dataroot)

    # Construct train and test transform operations
    transform_train = Compose([
        ToTensor(),
    ])
    transform_test = Compose([
        ToTensor(),
    ])

    # PyTorch Dataset classes for train, validation and test sets
    train_dataset = O2P2Dataset(phys_net_data.train, transform=transform_train)
    val_dataset = O2P2Dataset(phys_net_data.val, transform=transform_test)
    test_dataset = O2P2Dataset(phys_net_data.test, transform=transform_test)

    # PyTorch Dataloaders for train, validation and test sets
    train_loader = DataLoader(train_dataset, batch_size=opt.train_batch_size, shuffle=True, pin_memory=use_gpu)
    val_loader = DataLoader(val_dataset, batch_size=opt.test_batch_size, shuffle=False, pin_memory=use_gpu)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, pin_memory=use_gpu)

    # Initialize model
    percept = Percept()
    physics = Physics()
    render = Render()
    if use_gpu:
        percept.cuda()
        physics.cuda()
        render.cuda()

    # Define loss and optimizer
    # TODO: Different losses will be used to optimize different modules
    criterion = torch.nn.MSELoss()
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

        # compute model output
        objects = []
        for seg in segs:
            # TODO: objects from segs can be computed as a single batch
            obj = percept(seg)
            objects.append(obj)

        img0_reconstruction = render(objects)
        objects_evolved = physics(objects)
        img1_reconstruction = render(objects_evolved)

        # measure and record loss
        # TODO: Perceptual loss with VGG will be added
        percept_loss = criterion(img0, img0_reconstruction)
        physics_loss = criterion(img1, img1_reconstruction)
        render_loss = percept_loss + physics_loss

        # compute gradient and do optimizer step
        # TODO: Different losses will be used to optimize different modules
        optimizer.zero_grad()
        render_loss.backward()
        optimizer.step()

        return percept_loss, physics_loss, render_loss


if __name__ == '__main__':
    main()

