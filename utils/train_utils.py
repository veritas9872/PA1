import torch
from torch import nn, optim
from pathlib import Path


class CheckpointManager:
    """
    A checkpoint manager for Pytorch models and optimizers loosely based on Keras/Tensorflow Checkpointers.
    I should note that I am not sure whether this works in Pytorch graph mode.
    Giving up on saving as HDF5 files like in Keras. Just too annoying.
    Note that the whole system is based on 1 indexing, not 0 indexing.
    """
    def __init__(self, model, optimizer=None, mode='min', save_best_only=True, ckpt_dir='./checkpoints', max_to_keep=5):

        # Type checking.
        assert isinstance(model, nn.Module), 'Not a Pytorch Model'
        assert optimizer is None or isinstance(optimizer, optim.Optimizer), 'Not a Pytorch Optimizer'
        assert isinstance(max_to_keep, int) and (max_to_keep >= 0), 'Not a non-negative integer'
        assert mode in ('min', 'max'), 'Mode must be either `min` or `max`'
        ckpt_path = Path(ckpt_dir)
        assert ckpt_path.exists(), 'Not a valid, existing path'

        record_path = ckpt_path / 'Checkpoints.txt'

        try:
            record_file = open(record_path, mode='x')
        except FileExistsError:
            import sys
            print('WARNING: It is recommended to have a separate checkpoint directory for each run.', file=sys.stderr)
            print('Appending to previous Checkpoint record file!', file=sys.stderr)
            record_file = open(record_path, mode='a')

        print(f'Checkpoint List for {ckpt_path}', file=record_file)
        record_file.close()

        self.model = model
        self.optimizer = optimizer
        self.save_best_only = save_best_only
        self.ckpt_path = ckpt_path
        self.max_to_keep = max_to_keep
        self.save_counter = 0
        self.record_path = record_path
        self.record_dict = dict()

        if mode == 'min':
            self.prev_best = float('inf')
            self.mode = mode
        elif mode == 'max':
            self.prev_best = -float('inf')
            self.mode = mode
        else:
            raise TypeError('Mode must be either `min` or `max`')

    def _save(self, ckpt_name=None, **save_args):
        self.save_counter += 1
        save_dict = {'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
        save_dict.update(save_args)
        save_path = self.ckpt_path / (f'{ckpt_name}.tar' if ckpt_name else f'ckpt_{self.save_counter:03d}.tar')

        torch.save(save_dict, save_path)
        print(f'Saved Checkpoint to {save_path}')
        print(f'Checkpoint {self.save_counter:04d}: {save_path}')

        with open(file=self.record_path, mode='a') as file:
            print(f'Checkpoint {self.save_counter:04d}: {save_path}', file=file)

        self.record_dict[self.save_counter] = save_path

        if self.save_counter > self.max_to_keep:
            for count, ckpt_path in self.record_dict.items():  # This system uses 1 indexing.
                if (count <= (self.save_counter - self.max_to_keep)) and ckpt_path.exists():
                    ckpt_path.unlink()  # Delete existing checkpoint

        return save_path

    def save(self, metric, verbose=True, ckpt_name=None, **save_args):  # save_args are extra variables to save
        if self.mode == 'min':
            is_best = metric < self.prev_best
        elif self.mode == 'max':
            is_best = metric > self.prev_best
        else:
            raise TypeError('Mode must be either `min` or `max`')

        save_path = None
        if is_best or not self.save_best_only:
            save_path = self._save(ckpt_name)

        if verbose:
            if is_best:
                print(f'Metric improved from {self.prev_best:.4e} to {metric:.4e}')
            else:
                print(f'Metric did not improve.')

            if save_path:
                print(f'Saving Checkpoint at {save_path}')
            else:
                print('Not saving checkpoint.')

        if is_best:  # Update new best metric.
            self.prev_best = metric

        # Returns where the file was saved if any was saved. Also returns whether this was the best on the metric.
        return save_path, is_best  # So that one can see whether this one is the best or not.

    def load(self, load_dir, load_optimizer=True):
        save_dict = torch.load(load_dir)

        self.model.load_state_dict(save_dict['model_state_dict'])

        if load_optimizer:
            self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])

    def load_latest(self, load_root):
        pass
