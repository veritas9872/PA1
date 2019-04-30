import torch
import torchvision

from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from pathlib import Path
from time import time

from utils.run_utils import initialize, get_logger


def main():

    batch_size = 8
    num_workers = 4
    init_lr = 2E-4
    gpu = 1  # Set to None for CPU mode.
    num_epochs = 500
    verbose = False
    save_best_only = True

    data_root = '/home/veritas/PycharmProjects/PA1/data'
    log_root = '/home/veritas/PycharmProjects/PA1/logs'
    ckpt_root = '/home/veritas/PycharmProjects/PA1/checkpoints'

    ckpt_path = Path(ckpt_root)
    ckpt_path.mkdir(exist_ok=True)

    run_number, run_name = initialize(ckpt_path)

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
    model = torchvision.models.resnet50(pretrained=False, num_classes=100).to(device, non_blocking=True)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    # No softmax layer at the end necessary. Just need logits.
    loss_func = CrossEntropyLoss().to(device, non_blocking=True)

    # Getting data collection
    train_loss_sum = torch.as_tensor(0.)
    val_loss_sum = torch.as_tensor(0.)

    train_top1_correct = torch.as_tensor(0)
    val_top1_correct = torch.as_tensor(0)

    train_top5_correct = torch.as_tensor(0)
    val_top5_correct = torch.as_tensor(0)

    previous_best = 0.

    # Training loop. Please excuse my use of 1 based indexing here.
    logger.info('Beginning Training loop')
    for epoch in range(1, num_epochs+1):
        tic = time()
        model.train()
        torch.autograd.set_grad_enabled(True)
        for idx, (images, labels) in enumerate(train_loader, start=1):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(images)
            # Pytorch uses (input, target) ordering. Their shapes are also different.
            step_loss = loss_func(preds, labels)
            step_loss.backward()
            optimizer.step()

            with torch.no_grad():  # Necessary for metric calculation without errors due to gradient accumulation.
                train_loss_sum += step_loss
                top1 = torch.argmax(preds, dim=1)
                _, top5 = torch.topk(preds, k=5)  # Get only the indices.
                train_top1_correct += torch.sum(torch.eq(top1, labels))
                train_top5_correct += torch.sum(torch.eq(top5, labels.unsqueeze(-1)))  # This is probably right.

                if verbose:
                    print(f'Training loss Epoch {epoch:03d} Step {idx:03d}: {step_loss.item()}')

        else:
            toc = int(time() - tic)
            # Last step with small batch causes some inaccuracy but that is tolerable.
            epoch_loss = train_loss_sum.item() * batch_size / len(train_loader.dataset)
            epoch_top1_acc = train_top1_correct.item() / len(train_loader.dataset) * 100
            epoch_top5_acc = train_top5_correct.item() / len(train_loader.dataset) * 100

            msg = f'loss: {epoch_loss:.4f}, top1 accuracy: {epoch_top1_acc:.2f}%, top5 accuracy: {epoch_top5_acc:.2f}%'
            logger.info(f'Epoch {epoch:03d} Training. {msg} Time: {toc}s')

            # writer.add_scalar('train_loss', epoch_loss, global_step=epoch)
            # writer.add_scalar('train_acc', epoch_top1_acc, global_step=epoch)

            # Reset to 0. There must be a better way though...
            train_loss_sum = torch.as_tensor(0.)
            train_top1_correct = torch.as_tensor(0)
            train_top5_correct = torch.as_tensor(0)

        tic = time()
        model.eval()
        torch.autograd.set_grad_enabled(False)
        for idx, (images, labels) in enumerate(val_loader, start=1):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(images)

            top1 = torch.argmax(preds, dim=1)
            _, top5 = torch.topk(preds, k=5)

            val_top1_correct += torch.sum(torch.eq(top1, labels))
            val_top5_correct += torch.sum(torch.eq(top5, labels.unsqueeze(-1)))

            step_loss = loss_func(preds, labels)
            val_loss_sum += step_loss

            if verbose:
                print(f'Validation loss Epoch {epoch:03d} Step {idx:03d}: {step_loss.item()}')

        else:
            toc = int(time() - tic)
            epoch_loss = val_loss_sum.item() * batch_size / len(val_loader.dataset)
            epoch_top1_acc = val_top1_correct.item() / len(val_loader.dataset) * 100
            epoch_top5_acc = val_top5_correct.item() / len(val_loader.dataset) * 100

            msg = f'loss: {epoch_loss:.4f}, top1 accuracy: {epoch_top1_acc:.2f}%, top5 accuracy: {epoch_top5_acc:.2f}%'
            logger.info(f'Epoch {epoch:03d} Validation. {msg} Time: {toc}s')

            # writer.add_scalar('val_loss', epoch_loss, global_step=epoch)
            # writer.add_scalar('val_acc', epoch_top1_acc, global_step=epoch)

            # Reset to 0. There must be a better way though...
            val_loss_sum = torch.as_tensor(0.)
            val_top1_correct = torch.as_tensor(0)
            val_top5_correct = torch.as_tensor(0)

        # Checkpoint generation. Only implemented for single GPU models, not multi-gpu models.
        # All comparisons are done with python numbers, not tensors.
        if epoch_top5_acc > previous_best:  # Assumes larger metric is better.
            logger.info(f'Top 5 Validation Accuracy in Epoch {epoch} has improved from '
                        f'{previous_best:.2f}% to {epoch_top5_acc:.2f}%')
            previous_best = epoch_top5_acc

            save_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            torch.save(save_dict, ckpt_path / f'epoch_{epoch:03d}.tar')

        else:
            logger.info(f'Top 5 Validation Accuracy in Epoch {epoch} has not improved from the previous best epoch')

            if not save_best_only:
                save_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                torch.save(save_dict, ckpt_path / f'epoch_{epoch:03d}.tar')


if __name__ == '__main__':
    main()
