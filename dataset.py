"""Classes and functions to build and transform datasets."""

import random
import multiprocessing
import time
import threading
import queue
import torch

from data import encode_line


def _batch_elements(elements):
    if not elements:
        return elements
    if isinstance(elements[0], tuple):
        return tuple(list(batch) for batch in zip(*elements))
    raise TypeError("Cannot batch element")


class TextFileDataset:
    """Read lines from a text dataset."""

    def __init__(self, path):
        self._path = path

    def __iter__(self):
        with open(self._path) as f:
            for line in f:
                yield line.rstrip("\r\n")


class ZipDataset:
    """Read elements from parallel datasets."""

    def __init__(self, *datasets):
        self._datasets = datasets

    def __iter__(self):
        for elements in zip(*self._datasets):
            yield elements


class RepeatDataset:
    """Repeat a dataset."""

    def __init__(self, dataset, num_repeats=None):
        self._dataset = dataset
        self._num_repeats = num_repeats

    def __iter__(self):
        if self._num_repeats is None:
            while True:
                yield from iter(self._dataset)

        else:
            for _ in range(self._num_repeats):
                yield from iter(self._dataset)


class GroupDataset:
    """Group consecutive dataset elements."""

    def __init__(self, dataset, group_size):
        self._dataset = dataset
        self._group_size = group_size

    def __iter__(self):
        group = []

        for batch in self._dataset:
            group.append(batch)

            if len(group) == self._group_size:
                yield group
                group = []


class ShardDataset:
    """Read a subset of a dataset."""

    def __init__(self, dataset, num_shards, shard_index):
        self._dataset = dataset
        self._num_shards = num_shards
        self._shard_index = shard_index

    def __iter__(self):
        for i, element in enumerate(self._dataset):
            if i % self._num_shards == self._shard_index:
                yield element


class ShuffleDataset:
    """Read dataset elements in a random order."""

    def __init__(self, dataset, buffer_size=None):
        self._dataset = dataset
        self._buffer_size = buffer_size

    def _shuffle_and_yield(self, elements):
        print("Shuffling %d elements" % len(elements))
        random.shuffle(elements)
        while elements:
            yield elements.pop()

    def __iter__(self):
        elements = []

        for element in self._dataset:
            elements.append(element)

            if self._buffer_size is not None and len(elements) == self._buffer_size:
                yield from self._shuffle_and_yield(elements)

        if elements:
            yield from self._shuffle_and_yield(elements)


class MapDataset:
    """Apply a transformation on dataset elements."""

    def __init__(self, dataset, map_fn):
        self._dataset = dataset
        self._map_fn = map_fn

    def __iter__(self):
        for element in self._dataset:
            yield self._map_fn(element)


class FilterDataset:
    """Keep dataset elements that satisfy a condition."""

    def __init__(self, dataset, filter_fn):
        self._dataset = dataset
        self._filter_fn = filter_fn

    def __iter__(self):
        for element in self._dataset:
            if self._filter_fn(element):
                yield element


class BatchDataset:
    """Batch a dataset by the number of elements."""

    def __init__(self, dataset, batch_size, drop_remainder=False):
        self._dataset = dataset
        self._batch_size = batch_size
        self._drop_remainder = drop_remainder

    def __iter__(self):
        batch = []

        for element in self._dataset:
            batch.append(element)

            if len(batch) == self._batch_size:
                yield _batch_elements(batch)
                batch = []

        if not self._drop_remainder and batch:
            yield _batch_elements(batch)


class BatchByTokensDataset:
    """Batch a dataset by the number of tokens."""

    def __init__(
        self,
        dataset,
        batch_size,
        length_fn,
        length_bucket_width,
        maximum_length,
        drop_remainder=False,
    ):
        self._dataset = dataset
        self._length_fn = length_fn
        self._length_bucket_width = length_bucket_width
        self._drop_remainder = drop_remainder

        self._max_length_per_bucket = list(
            range(length_bucket_width, maximum_length + 1, length_bucket_width)
        )
        if self._max_length_per_bucket[-1] != maximum_length:
            self._max_length_per_bucket.append(maximum_length)

        self._batch_size_per_bucket = [
            max(batch_size // max_len, 1) for max_len in self._max_length_per_bucket
        ]

        # Reduce batch to a multiple of 8 to enable NVIDIA Tensor Cores.
        self._batch_size_per_bucket = [
            max(batch_size - batch_size % 8, 1)
            for batch_size in self._batch_size_per_bucket
        ]

    def _get_bucket_id(self, length):
        for i, max_length in enumerate(self._max_length_per_bucket):
            if max_length - self._length_bucket_width < length <= max_length:
                return i

    def __iter__(self):
        buckets = [[] for _ in self._max_length_per_bucket]

        for element in self._dataset:
            length = self._length_fn(element)
            bucket_id = self._get_bucket_id(length)
            bucket = buckets[bucket_id]
            bucket.append(element)

            if len(bucket) == self._batch_size_per_bucket[bucket_id]:
                yield _batch_elements(bucket)
                buckets[bucket_id] = []

        if not self._drop_remainder:
            for bucket in self._buckets:
                if bucket:
                    yield _batch_elements(bucket)


class PrefetchDataset:
    """Prefetch dataset elements in a background process or thread."""

    def __init__(self, dataset, prefetch_size=1, use_threading=False):
        self._dataset = dataset
        self._prefetch_size = prefetch_size
        self._use_threading = use_threading

    def _fetch_elements(self, queue):
        for element in self._dataset:
            queue.put(element)
        queue.put(None)

    def __iter__(self):
        if self._use_threading:
            queue_cls = queue.Queue
            worker_cls = threading.Thread
        else:
            context = multiprocessing.get_context("spawn")
            queue_cls = context.Queue
            worker_cls = context.Process

        producer_queue = queue_cls(self._prefetch_size)
        producer = worker_cls(
            target=self._fetch_elements, args=(producer_queue,), daemon=True
        )
        producer.start()

        while True:
            element = producer_queue.get()
            if element is None:
                break
            yield element

        producer.join()


class LatencyDataset:
    """Dataset wrapper to compute the latency to get an element from the dataset."""

    def __init__(self, dataset, ignore_first_n=1):
        self._dataset = dataset
        self._avg_latency_us = 0
        self._num_samples = 0
        self._ignore_first_n = ignore_first_n

    @property
    def average_latency_us(self):
        return self._avg_latency_us

    def __iter__(self):
        iterator = iter(self._dataset)

        while True:
            try:
                start = time.time_ns()
                element = next(iterator)
                end = time.time_ns()

                if self._ignore_first_n > 0:
                    self._ignore_first_n -= 1
                else:
                    latency_us = (end - start) / 1000
                    self._avg_latency_us = (
                        self._avg_latency_us * self._num_samples + latency_us
                    ) / (self._num_samples + 1)
                    self._num_samples += 1

                yield element
            except StopIteration:
                break


class EncodeTokens:
    """Transformation to encode text lines into a list of token IDs."""

    def __init__(self, source_vocabulary, target_vocabulary):
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary

    def __call__(self, element):
        source, target = element

        source = encode_line(source, self.source_vocabulary, add_eos=True)
        target = encode_line(target, self.target_vocabulary, add_bos=True, add_eos=True)

        if target:
            target_in = target[:-1]
            target_out = target[1:]
        else:
            target_in = []
            target_out = []

        return source, target_in, target_out


class FilterByLength:
    """Filter condition to keep elements satisfying the length constraints."""

    def __init__(self, maximum_source_length, maximum_target_length):
        self.maximum_source_length = maximum_source_length
        self.maximum_target_length = maximum_target_length

    def __call__(self, element):
        source, target, _ = element
        return (
            0 < len(source) <= self.maximum_source_length
            and 0 < len(target) <= self.maximum_target_length
        )


def length_fn(element):
    """Returns the representative length for a parallel source/target example."""
    source, target, _ = element
    return max(len(source), len(target))


class ConvertToTensor:
    """Transformation to convert Python lists to PyTorch tensors."""

    def __init__(self, device, padding_idx=0, pad_to_multiple=1):
        self.device = device
        self.padding_idx = padding_idx
        self.pad_to_multiple = pad_to_multiple

    def __call__(self, elements):
        return tuple(
            to_tensor(
                element,
                device=self.device,
                padding_value=self.padding_idx,
                pad_to_multiple=self.pad_to_multiple,
            )
            for element in elements
        )


def to_tensor(batch_ids, device=None, padding_value=0, pad_to_multiple=1):
    """Converts a batch of token IDs into a dense 2D tensor."""
    maximum_length = max(len(ids) for ids in batch_ids)
    if maximum_length % pad_to_multiple != 0:
        maximum_length += pad_to_multiple - (maximum_length % pad_to_multiple)

    batch_ids = [
        ids + [padding_value] * (maximum_length - len(ids)) for ids in batch_ids
    ]

    return torch.tensor(batch_ids, device=device)


def create_training_dataset(
    source_dataset,
    target_dataset,
    source_vocabulary,
    target_vocabulary,
    batch_size=4096,
    batch_type="tokens",
    pad_to_multiple=1,
    padding_idx=0,
    maximum_source_length=150,
    maximum_target_length=150,
    num_accum_batches=None,
    device="cpu",
    num_shards=1,
    shard_index=0,
    prefetch_size=None,
    shuffle_buffer_size=None,
):
    """Creates a dataset with all transformations required for training."""

    if isinstance(source_dataset, str):
        source_dataset = TextFileDataset(source_dataset)
    if isinstance(target_dataset, str):
        target_dataset = TextFileDataset(target_dataset)

    dataset = ZipDataset(source_dataset, target_dataset)

    if num_shards > 1:
        dataset = ShardDataset(dataset, num_shards, shard_index)

    dataset = ShuffleDataset(dataset, shuffle_buffer_size)
    dataset = RepeatDataset(dataset)
    dataset = MapDataset(dataset, EncodeTokens(source_vocabulary, target_vocabulary))
    dataset = FilterDataset(
        dataset,
        FilterByLength(maximum_source_length, maximum_target_length),
    )

    if batch_type == "tokens":
        dataset = BatchByTokensDataset(
            dataset,
            batch_size=batch_size,
            length_fn=length_fn,
            length_bucket_width=pad_to_multiple,
            maximum_length=max(maximum_source_length, maximum_target_length),
            drop_remainder=True,
        )
    else:
        dataset = BatchDataset(dataset, batch_size, drop_remainder=True)

    if prefetch_size is None:
        prefetch_size = num_accum_batches if num_accum_batches is not None else 1

    # Prepare batches in a separate process for true parallelism,
    # then bufferize in a separate thread.
    dataset = PrefetchDataset(dataset, prefetch_size=prefetch_size, use_threading=False)
    dataset = MapDataset(dataset, ConvertToTensor(device, padding_idx, pad_to_multiple))
    dataset = PrefetchDataset(dataset, prefetch_size=prefetch_size, use_threading=True)

    if num_accum_batches is not None:
        dataset = GroupDataset(dataset, num_accum_batches)

    return dataset
