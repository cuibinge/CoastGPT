from typing import Dict, Iterator, List, Optional
from operator import itemgetter
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

# 分布式采样器包装类，用于在分布式训练中使用任意采样器
class DistributedSamplerWrapper(DistributedSampler):
    """
    用于分布式训练的 `Sampler` 包装器。
    允许在分布式模式下使用任意采样器。

    当与 `torch.nn.parallel.DistributedDataParallel` 结合使用时特别有用。
    在这种情况下，每个进程可以将一个 DistributedSamplerWrapper 实例作为 DataLoader 的采样器，
    并加载原始数据集的一个子采样子集，且该子集是该进程独有的。

    注意：
        假设采样器的大小是固定的。
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        参数:
            sampler: 用于子采样的采样器
            num_replicas (int, 可选): 参与分布式训练的进程数量
            rank (int, 可选): 当前进程在 `num_replicas` 中的排名
            shuffle (bool, 可选): 如果为 True（默认），采样器将对索引进行洗牌
        """
        # 调用父类的初始化方法，传入从采样器创建的数据集
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        # 保存传入的采样器
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """
        迭代采样器。

        返回:
            Python 迭代器
        """
        # 从采样器创建数据集
        self.dataset = DatasetFromSampler(self.sampler)
        # 调用父类的 __iter__ 方法获取索引的索引
        indexes_of_indexes = super().__iter__()
        # 获取子采样器的索引
        subsampler_indexes = self.dataset
        # 通过 itemgetter 函数根据索引的索引获取对应的子采样器索引，并返回迭代器
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

# 从采样器创建数据集的类
class DatasetFromSampler(Dataset):
    """
    从 `Sampler` 创建索引的数据集。

    参数:
        sampler: PyTorch 采样器
    """

    def __init__(self, sampler: Sampler):
        """
        DatasetFromSampler 的初始化方法。
        """
        # 保存传入的采样器
        self.sampler = sampler
        # 初始化采样器列表为 None
        self.sampler_list = None

    def __getitem__(self, index: int):
        """
        获取数据集中的元素。

        参数:
            index: 数据集中元素的索引

        返回:
            根据索引获取的单个元素
        """
        # 如果采样器列表未初始化
        if self.sampler_list is None:
            # 将采样器转换为列表
            self.sampler_list = list(self.sampler)
        # 根据索引返回采样器列表中的元素
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        返回:
            int: 数据集的长度
        """
        # 返回采样器的长度
        return len(self.sampler)

# 自定义的整理函数
def collate_func(input):
    # 直接返回输入列表的第一个元素
    return input[0]

# 无限重复的采样器类
class _RepeatSampler(object):
    """
    永远重复的采样器。
    参数:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        # 保存传入的采样器
        self.sampler = sampler

    def __iter__(self):
        # 无限循环
        while True:
            # 不断迭代采样器
            yield from iter(self.sampler)

# 无限数据加载器类
class InfiniteDataLoader(DataLoader):
    """
    重用工作进程的数据加载器。
    使用与普通 DataLoader 相同的语法。
    """

    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 将批采样器替换为无限重复的采样器
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        # 获取父类的迭代器
        self.iterator = super().__iter__()

    def __len__(self):
        # 返回批采样器中采样器的长度
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        # 循环数据集长度的次数
        for i in range(len(self)):
            # 从迭代器中获取下一个元素并返回
            yield next(self.iterator)