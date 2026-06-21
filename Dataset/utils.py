from typing import Dict, Iterator, List, Optional
from operator import itemgetter
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

class DistributedSamplerWrapper(DistributedSampler):
    """
    genericㄤgenericgeneric `Sampler` genericgenericㄣ€?
    genericgenericㄥgeneric″genericㄤgeneric?

    generic `torch.nn.parallel.DistributedDataParallel` genericgenericㄣ€?
    genericㄨgenericgenericュgeneric€generic?DistributedSamplerWrapper generic DataLoader generic?
    genericgenericgenericュgenericヨgeneric?

    genericㄦgeneric?
        genericgenericㄧgenericуgenericgeneric?
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        generic:
            sampler: genericㄤgeneric?
            num_replicas (int, generic€?: genericgeneric▼generic
            rank (int, generic€?: generic▼generic?`num_replicas` genericgeneric
            shuffle (bool, generic€?: generic?Truegenericわgenericgeneric㈠genericgeneric
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """
        genericgenericㄣ€?

        generic:
            Python genericgeneric?
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

class DatasetFromSampler(Dataset):
    """
    generic?`Sampler` generic㈠genericgeneric?

    generic:
        sampler: PyTorch generic?
    """

    def __init__(self, sampler: Sampler):
        """
        DatasetFromSampler generic?
        """
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """
        generic€?

        generic:
            index: generic?

        generic:
            generic㈠genericgeneric?
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        generic:
            int: generic
        """
        return len(self.sampler)

def collate_func(input):
    return input[0]

class _RepeatSampler(object):
    """
    genericgeneric?
    generic:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class InfiniteDataLoader(DataLoader):
    """
    genericヤgeneric▼genericgeneric?
    generic?DataLoader genericgeneric€?
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
