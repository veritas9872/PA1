import torch
import torchvision
from tensorboardX import SummaryWriter

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from time import time
from pathlib import Path

from utils.run_utils import initialize, get_logger
from utils.train_utils import CheckpointManager
from train.training import train_epoch, eval_epoch
from models.senet import se_resnet50_cifar100


def main():

    # Put these in args later.
    batch_size = 12
    num_workers = 8
    init_lr = 2E-4
    gpu = 0  # Set to None for CPU mode.
    num_epochs = 500
    verbose = False
    save_best_only = True
    max_to_keep = 100
    data_root = '/home/veritas/PycharmProjects/PA1/data'
    ckpt_root = '/home/veritas/PycharmProjects/PA1/checkpoints'
    log_root = '/home/veritas/PycharmProjects/PA1/logs'

    # Beginning session.
    run_number, run_name = initialize(ckpt_root)

    ckpt_path = Path(ckpt_root)
    ckpt_path.mkdir(exist_ok=True)
    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    log_path = Path(log_root)
    log_path.mkdir(exist_ok=True)
    log_path = log_path / run_name
    log_path.mkdir(exist_ok=True)

    logger = get_logger(name=__name__, save_file=log_path / run_name)

    if (gpu is not None) and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cpu')

    # Do more fancy transforms later.
    transform = torchvision.transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR100(data_root, train=True, transform=transform, download=True)
    val_dataset = torchvision.datasets.CIFAR100(data_root, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Define model, optimizer, etc.
    model = se_resnet50_cifar100().to(device, non_blocking=True)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    # No softmax layer at the end necessary. Just need logits.
    loss_func = nn.CrossEntropyLoss().to(device, non_blocking=True)

    # Create checkpoint manager
    checkpointer = CheckpointManager(model, optimizer, ckpt_path, save_best_only, max_to_keep)

    # For recording data.
    previous_best = 0.  # Accuracy should improve.

    # Training loop. Please excuse my use of 1 based indexing here.
    logger.info('Beginning Training loop')
    for epoch in range(1, num_epochs+1):
        # Start of training
        tic = time()
        train_loss_sum, train_top1_correct, train_top5_correct = \
            train_epoch(model, optimizer, loss_func, train_loader, device, epoch, verbose)

        toc = int(time() - tic)
        # Last step with small batch causes some inaccuracy but that is tolerable.
        train_epoch_loss = train_loss_sum.item() * batch_size / len(train_loader.dataset)
        train_epoch_top1_acc = train_top1_correct.item() / len(train_loader.dataset) * 100
        train_epoch_top5_acc = train_top5_correct.item() / len(train_loader.dataset) * 100

        msg = f'Epoch {epoch:03d} Training. loss: {train_epoch_loss:.4f}, ' \
            f'top1 accuracy: {train_epoch_top1_acc:.2f}%, top5 accuracy: {train_epoch_top5_acc:.2f}% Time: {toc}s'
        logger.info(msg)

        # Start of evaluation
        tic = time()
        val_loss_sum, val_top1_correct, val_top5_correct = \
            eval_epoch(model, loss_func, val_loader, device, epoch, verbose)

        toc = int(time() - tic)
        val_epoch_loss = val_loss_sum.item() * batch_size / len(val_loader.dataset)
        val_epoch_top1_acc = val_top1_correct.item() / len(val_loader.dataset) * 100
        val_epoch_top5_acc = val_top5_correct.item() / len(val_loader.dataset) * 100

        msg = f'Epoch {epoch:03d} Validation. loss: {val_epoch_loss:.4f}, ' \
            f'top1 accuracy: {val_epoch_top1_acc:.2f}%, top5 accuracy: {val_epoch_top5_acc:.2f}%  Time: {toc}s'
        logger.info(msg)

        if val_epoch_top5_acc > previous_best:  # Assumes larger metric is better.
            logger.info(f'Top 5 Validation Accuracy in Epoch {epoch} has improved from '
                        f'{previous_best:.2f}% to {val_epoch_top5_acc:.2f}%')
            previous_best = val_epoch_top5_acc
            checkpointer.save(is_best=True)

        else:
            logger.info(f'Top 5 Validation Accuracy in Epoch {epoch} has not improved from the previous best epoch')
            checkpointer.save(is_best=False)


if __name__ == '__main__':
    main()
