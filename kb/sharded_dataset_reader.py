import glob
import random

from typing import Iterator

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from os.path import basename


@DatasetReader.register('sharded')
class ShardedDatasetReader(DatasetReader):
    """
    Wraps another dataset reader and uses it to read from multiple input files
    using a single process. Note that in this case the ``file_path`` passed to
    ``read()`` should be a glob, and that the dataset reader will return
    instances from all files matching the glob.

    Parameters
    ----------
    base_reader : ``DatasetReader``
        Each process will use this dataset reader to read zero or more files.
    buffer_size: ``int``, (optional, default=1000)
        The size of the buffer on which read instances are placed to be
        yielded. Uses more memory but disk access is more efficient.
    """

    def __init__(self,
                 base_reader: DatasetReader,
                 buffer_size: int = 1000,
                 shuffle: bool = True) -> None:
        # Sharded reader is intrinsically lazy.
        super().__init__(lazy=True)
        self.reader = base_reader
        self.buffer_size = buffer_size
        self.shuffle = shuffle

    def text_to_instance(self, *args, **kwargs) -> Instance:
        """
        Just delegate to the base reader text_to_instance.
        """
        return self.reader.text_to_instance(*args, **kwargs)

    def _read(self, file_path: str) -> Iterator[Instance]:
        """
        A generator that reads instances off the files successively and
        yields them up until none are left.
        """
        shards = glob.glob(file_path)
        if self.shuffle:
            random.shuffle(shards)
        else:
            shards.sort()
        buffer = []
        for file in shards:
            print(f"Starting processing of {basename(file)}.", flush=True)
            # If base reader is lazy, reader.read() returns a _LazyInstances
            # (defined in the same file as AllenNLP DatasetReader). Calling
            # iter() gives us an object we can actually iterate over.
            # If base reader is not lazy, reader.read() is already iterable
            # directly but iter() doesn't hurt anything.
            instance_gen = iter(self.reader.read(file))
            file_is_empty = False
            while not file_is_empty:
                while len(buffer) < self.buffer_size:
                    try:
                        buffer.append(next(instance_gen))
                    except StopIteration:
                        file_is_empty = True
                        break
                if self.shuffle:
                    random.shuffle(buffer)
                for instance in buffer:
                    yield instance
                buffer.clear()
