import math
import os
import logging
import random

import numpy as np
import torch

from fairseq.data.iterators import CountingIterator, BufferedIterator
from fairseq.data import data_utils

from one_peace.utils.data_utils import new_islice

logger = logging.getLogger(__name__)


class EpochBatchIterator:
    """A multi-epoch iterator over a :class:`torch.utils.data.Dataset`.

    Compared to :class:`torch.utils.data.DataLoader`, this iterator:

    - can be reused across multiple epochs with the :func:`next_epoch_itr`
      method (optionally shuffled between epochs)
    - can be serialized/deserialized with the :func:`state_dict` and
      :func:`load_state_dict` methods
    - supports sharding with the *num_shards* and *shard_id* arguments

    Args:
        dataset (~torch.utils.data.Dataset): dataset from which to load the data
        collate_fn (callable): merges a list of samples to form a mini-batch
        batch_sampler (~torch.utils.data.Sampler or a callable): an iterator over batches of
            indices, or a callable to create such an iterator (~torch.utils.data.Sampler).
            A callable batch_sampler will be called for each epoch to enable per epoch dynamic
            batch iterators defined by this callable batch_sampler.
        seed (int, optional): seed for random number generator for
            reproducibility (default: 1).
        num_shards (int, optional): shard the data iterator into N
            shards (default: 1).
        shard_id (int, optional): which shard of the data iterator to
            return (default: 0).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means the data will be loaded in the main process
            (default: 0).
        epoch (int, optional): the epoch to start the iterator from
            (default: 1).
        buffer_size (int, optional): the number of batches to keep ready in the
            queue. Helps speeding up dataloading. When buffer_size is zero, the
            default torch.utils.data.DataLoader preloading is used.
        timeout (int, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative (default: ``0``).
        disable_shuffling (bool, optional): force disable shuffling
            (default: ``False``).
    """

    def __init__(
        self,
        dataset,
        collate_fn,
        batch_sampler,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        buffer_size=0,
        timeout=0,
        disable_shuffling=False,
        persistent_workers=False
    ):
        assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.batch_sampler = batch_sampler
        self._frozen_batches = (
            tuple(batch_sampler) if not callable(batch_sampler) else None
        )
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.num_workers = num_workers
        # This upper limit here is to prevent people from abusing this feature
        # in a shared computing environment.
        self.buffer_size = min(buffer_size, 20)
        self.timeout = timeout
        self.disable_shuffling = disable_shuffling
        self.persistent_workers = persistent_workers

        self.epoch = max(epoch, 1)  # we use 1-based indexing for epochs
        self.shuffle = not disable_shuffling
        self._cur_epoch_itr = None
        self._next_epoch_itr = None

    @property
    def frozen_batches(self):
        if self._frozen_batches is None:
            self._frozen_batches = tuple(self.batch_sampler(self.dataset, self.epoch))
        return self._frozen_batches

    @property
    def first_batch(self):
        if len(self.frozen_batches) == 0:
            raise Exception(
                "The dataset is empty. This could indicate "
                "that all elements in the dataset have been skipped. "
                "Try increasing the max number of allowed tokens or using "
                "a larger dataset."
            )

        if getattr(self.dataset, "supports_fetch_outside_dataloader", True):
            return self.collate_fn([self.dataset[-1] for i in self.frozen_batches[0]])
        else:
            return "DUMMY"

    def __len__(self):
        return int(math.ceil(len(self.frozen_batches) / float(self.num_shards)))

    @property
    def n(self):
        return self.iterations_in_epoch

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called."""
        if self._next_epoch_itr is not None:
            return self.epoch
        elif self._cur_epoch_itr is not None and self.end_of_epoch():
            return self.epoch + 1
        else:
            return self.epoch

    def next_epoch_itr(
        self, shuffle=True, fix_batches_to_gpus=False, set_dataset_epoch=True
    ):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus (bool, optional): ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
            set_dataset_epoch (bool, optional): update the wrapped Dataset with
                the new epoch number (default: True).
        """
        if self.disable_shuffling:
            shuffle = False
        prev_epoch = self.epoch
        self.epoch = self.next_epoch_idx
        if set_dataset_epoch and hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.epoch)
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            if callable(self.batch_sampler) and prev_epoch != self.epoch:
                # reset _frozen_batches to refresh the next epoch
                self._frozen_batches = None
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch,
                shuffle,
                fix_batches_to_gpus=fix_batches_to_gpus,
            )
        self.shuffle = shuffle
        return self._cur_epoch_itr

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        """The number of consumed batches in the current epoch."""
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.n
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.n
        return 0

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        if self.end_of_epoch():
            epoch = self.epoch + 1
            iter_in_epoch = 0
        else:
            epoch = self.epoch
            iter_in_epoch = self.iterations_in_epoch
        return {
            "version": 2,
            "epoch": epoch,
            "iterations_in_epoch": iter_in_epoch,
            "shuffle": self.shuffle,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict["epoch"]
        itr_pos = state_dict.get("iterations_in_epoch", 0)
        version = state_dict.get("version", 1)
        if itr_pos > 0:
            # fast-forward epoch iterator
            self._next_epoch_itr = self._get_iterator_for_epoch(
                self.epoch,
                shuffle=state_dict.get("shuffle", True),
                offset=itr_pos,
            )
            if self._next_epoch_itr is None:
                if version == 1:
                    # legacy behavior: we finished the epoch, increment epoch counter
                    self.epoch += 1
                else:
                    raise RuntimeError(
                        "Cannot resume training due to dataloader mismatch, please "
                        "report this to the fairseq developers. You can relaunch "
                        "training with `--reset-dataloader` and it should work."
                    )
        else:
            self._next_epoch_itr = None

    def _get_iterator_for_epoch(
        self, epoch, shuffle, fix_batches_to_gpus=False, offset=0
    ):
        def shuffle_batches(batches, seed):
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
            return batches

        def worker_init(worked_id):
            worker_seed = torch.initial_seed() % 2 ** 32 + worked_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # if shuffle:
        #     batches = shuffle_batches(list(self.frozen_batches), self.seed + epoch)
        # else:
        #     batches = self.frozen_batches
        batches = self.frozen_batches

        sharded_len = int(math.ceil(len(batches) / float(self.num_shards)))
        batches = new_islice(batches, self.shard_id, len(batches), self.num_shards)
        if len(batches) < sharded_len:
            batches.append([])

        if offset > 0 and offset >= len(batches):
            return None

        if self.num_workers > 0:
            os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

        # Create data loader
        itr = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=batches[offset:],
            num_workers=self.num_workers,
            timeout=self.timeout,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            worker_init_fn=worker_init
        )

        # Wrap with a BufferedIterator if needed
        if self.buffer_size > 0:
            itr = BufferedIterator(self.buffer_size, itr)

        # Wrap with CountingIterator
        itr = CountingIterator(itr, start=offset)

        return itr