import torch
import torchvision
from time import time
from train.my_tranforms import train_transform, val_transform
from utils.run_utils import initialize, get_logger, save_dict_as_json
from utils.train_utils import CheckpointManager
from pathlib import Path
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def train_epoch(model, optimizer, loss_func, data_loader, device, epoch, verbose):
    model.train()
    torch.autograd.set_grad_enabled(True)

    # Reset to 0. There must be a better way though...
    loss_sum = torch.as_tensor(0.)
    top1_correct = torch.as_tensor(0.)
    top5_correct = torch.as_tensor(0.)

    for idx, (images, labels) in enumerate(data_loader, start=1):

        # I don't know where to put this for best optimization.
        images = images.to(device)
        labels = labels.to(device)

        step_loss, preds = train_step(model, optimizer, loss_func, images, labels)

        with torch.no_grad():  # Necessary for metric calculation without errors due to gradient accumulation.
            loss_sum += step_loss
            top1 = torch.argmax(preds, dim=1)
            _, top5 = torch.topk(preds, k=5)  # Get only the indices.
            top1_correct += torch.sum(torch.eq(top1, labels))
            top5_correct += torch.sum(torch.eq(top5, labels.unsqueeze(-1)))  # This is probably right.

            if verbose:
                print(f'Training loss Epoch {epoch:03d} Step {idx:03d}: {step_loss.item()}')

    return loss_sum, top1_correct, top5_correct


def train_step(model, optimizer, loss_func, images, labels):
    optimizer.zero_grad()
    preds = model(images)
    step_loss = loss_func(preds, labels)  # Pytorch uses (input, target) ordering. Their shapes are also different.
    step_loss.backward()
    optimizer.step()
    return step_loss, preds


def eval_epoch(model, loss_func, val_loader, device, epoch, verbose):
    model.eval()
    torch.autograd.set_grad_enabled(False)

    # Reset to 0. There must be a better way though...
    loss_sum = torch.as_tensor(0.)
    top1_correct = torch.as_tensor(0.)
    top5_correct = torch.as_tensor(0.)

    for idx, (images, labels) in enumerate(val_loader, start=1):

        # Not sure where this belongs.
        images = images.to(device)
        labels = labels.to(device)

        preds, top1, top5, step_loss = eval_step(model, loss_func, images, labels)

        loss_sum += step_loss
        top1_correct += torch.sum(torch.eq(top1, labels))
        top5_correct += torch.sum(torch.eq(top5, labels.unsqueeze(-1)))

        if verbose:
            print(f'Validation loss Epoch {epoch:03d} Step {idx:03d}: {step_loss.item()}')

    return loss_sum, top1_correct, top5_correct


def eval_step(model, loss_func, images, labels):
    preds = model(images)
    top1 = torch.argmax(preds, dim=1)
    _, top5 = torch.topk(preds, k=5)
    step_loss = loss_func(preds, labels)
    return preds, top1, top5, step_loss


def train_model(model, args):

    assert isinstance(model, nn.Module)

    # Beginning session.
    run_number, run_name = initialize(args.ckpt_root)

    ckpt_path = Path(args.ckpt_root)
    ckpt_path.mkdir(exist_ok=True)
    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    log_path = Path(args.log_root)
    log_path.mkdir(exist_ok=True)
    log_path = log_path / run_name
    log_path.mkdir(exist_ok=True)

    logger = get_logger(name=__name__, save_file=log_path / run_name)

    if (args.gpu is not None) and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    # Saving args for later use.
    save_dict_as_json(vars(args), log_dir=log_path, save_name=run_name)

    dataset_kwargs = dict(root=args.data_root, download=True)
    train_dataset = torchvision.datasets.CIFAR100(train=True, transform=train_transform(), **dataset_kwargs)
    val_dataset = torchvision.datasets.CIFAR100(train=False, transform=val_transform(), **dataset_kwargs)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Define model, optimizer, etc.
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # No softmax layer at the end necessary. Just need logits.
    loss_func = nn.CrossEntropyLoss(reduction='mean').to(device)

    # LR scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Create checkpoint manager
    checkpointer = CheckpointManager(model, optimizer, mode='max', save_best_only=args.save_best_only,
                                     ckpt_dir=ckpt_path, max_to_keep=args.max_to_keep)

    # Tensorboard Writer
    writer = SummaryWriter(log_dir=str(log_path))

    # Training loop. Please excuse my use of 1 based indexing here.
    logger.info('Beginning Training loop')
    for epoch in range(1, args.num_epochs + 1):
        # Start of training
        tic = time()
        train_loss_sum, train_top1_correct, train_top5_correct = \
            train_epoch(model, optimizer, loss_func, train_loader, device, epoch, args.verbose)

        toc = int(time() - tic)
        # Last step with small batch causes some inaccuracy but that is tolerable.
        train_epoch_loss = train_loss_sum.item() * args.batch_size / len(train_loader.dataset)
        train_epoch_top1_acc = train_top1_correct.item() / len(train_loader.dataset) * 100
        train_epoch_top5_acc = train_top5_correct.item() / len(train_loader.dataset) * 100

        logger.info(f'Epoch {epoch:03d} Training. loss: {train_epoch_loss:.4e}, top1 accuracy: '
                    f'{train_epoch_top1_acc:.2f}%, top5 accuracy: {train_epoch_top5_acc:.2f}% Time: {toc}s')

        # Writing to Tensorboard
        writer.add_scalar('train_epoch_loss', train_epoch_loss, epoch)
        writer.add_scalar('train_epoch_top1_acc', train_epoch_top1_acc, epoch)
        writer.add_scalar('train_epoch_top5_acc', train_epoch_top5_acc, epoch)

        # Start of evaluation
        tic = time()
        val_loss_sum, val_top1_correct, val_top5_correct = \
            eval_epoch(model, loss_func, val_loader, device, epoch, args.verbose)

        toc = int(time() - tic)
        val_epoch_loss = val_loss_sum.item() * args.batch_size / len(val_loader.dataset)
        val_epoch_top1_acc = val_top1_correct.item() / len(val_loader.dataset) * 100
        val_epoch_top5_acc = val_top5_correct.item() / len(val_loader.dataset) * 100

        logger.info(f'Epoch {epoch:03d} Validation. loss: {val_epoch_loss:.4e}, top1 accuracy: '
                    f'{val_epoch_top1_acc:.2f}%, top5 accuracy: {val_epoch_top5_acc:.2f}%  Time: {toc}s')

        # Writing to Tensorboard
        writer.add_scalar('val_epoch_loss', val_epoch_loss, epoch)
        writer.add_scalar('val_epoch_top1_acc', val_epoch_top1_acc, epoch)
        writer.add_scalar('val_epoch_top5_acc', val_epoch_top5_acc, epoch)
        for idx, group in enumerate(optimizer.param_groups, start=1):
            writer.add_scalar(f'learning_rate_{idx}', group['lr'], epoch)

        # Things to do after each epoch.
        scheduler.step()  # Reduces LR at the designated times. Probably does not use 1 indexing like me.
        checkpointer.save(metric=val_epoch_top5_acc)
