
__author__ = ['Michael Drews']

import numpy as np
from torch.utils.data import Dataset
from torch import Tensor
import sys
import tqdm
import h5py
import os
import time

import seg3d.fakedata as fakedata
import utils.volutils as volutils
import utils.misc as misc
from pathlib import Path

import traceback


def _percentile_normalization(image_data, percentiles=(1,99)):
    """Normalize pixel values within a given percentile range

    Args:
        image_data: input image data
        percentiles: tuple defining the percentiles

    Returns:
        clipped image data
    """
    low = np.percentile(image_data.flatten(), percentiles[0])
    high = np.percentile(image_data.flatten(), percentiles[1])


    image_data = (image_data-low) / (high-low)
    return np.clip(image_data, 0, 1)


class HDF5Dataset(Dataset):
    """
    Basic dataset for loading volumes from HDF5.
    Implements a memory which caches the last few requested chunks from the data.

    """

    def __init__(self, h5_files=None, chunk_size=64, mem_chunks=4, verbose=False, **kwargs):
        super().__init__(**kwargs)

        if type(h5_files) is str:
            h5_files = [h5_files]
        self.h5_files = h5_files

        # initialize cache
        self.data_cache = {}
        self.mem_chunks = mem_chunks
        self.chunk_size = chunk_size
        self.last_requests = []

        # get basic info about dataset
        self._get_files_info()

        self.verbose = verbose

    def _get_files_info(self):
        """
        Get lengths and keys of all dataset files.
        """

        file_keys = []
        file_lengths = []
        for i, filename in enumerate(self.h5_files):

            file_exists = os.path.isfile(filename)
            if not file_exists:
                print(f"Warning: File {filename} does not exist.")
                return None

            # if file exists
            with h5py.File(filename, 'r') as hf:

                # get keys of the internal dataset
                if i == 0:
                    file_keys = list(hf.keys())
                else:
                    # check if keys are the same for each files
                    assert set(list(hf.keys())) == set(file_keys), \
                        f'{filename} has a not the same keys as the other files.'

                # get sample number from each file
                lens = [hf[k].shape[0] for k in file_keys]
                assert len(set(lens)) == 1, f'{filename} has differently sized datasets.'
                file_lengths.append(lens[0])

        self.dataset_keys = file_keys
        self.file_lengths = file_lengths

    def __len__(self):
        return sum(self.file_lengths)

    def _idx2address(self, i):
        """
        Maps from sample index to 'address' given as a tuple of
        file number, chunk number, and local index
        """
        # get start index in each file
        file_start_i = np.cumsum(self.file_lengths)[:-1]
        file_start_i = np.insert(file_start_i, 0, 0)

        # find file number
        f_i = np.nonzero(i >= file_start_i)[0][-1]

        # within file, find chunk number
        i_ = i - file_start_i[f_i]
        c_i = i_ // self.chunk_size

        # get local index
        l_i = i_ % self.chunk_size

        return f_i, c_i, l_i

    @staticmethod
    def _transform_slice_to_indices(i):
        """
        Transforms a slice object to an explicit list of requested indices.
        """
        if type(i) is slice:
            if i.step is None:
                step = 1
            else:
                step = i.step
            idxs = list(range(i.start, i.stop, step))
        else:
            idxs = [i]
        return idxs

    @staticmethod
    def _transform_indices_to_slice(ind):
        """
        Transform a list of indices into a slice object.
        """
        if len(ind) > 1:
            start = ind[0]
            end = ind[-1]
            step = ind[1] - ind[0]
            return slice(start, end + 1, step)
        else:
            return slice(ind[0], ind[0] + 1, 1)

    def _clean_cache(self):
        """
        Reduces the cache size to the maximum allowed chunk number by
        deleting older chunks from the cache.
        """
        count = 0
        while len(self.data_cache) > self.mem_chunks:
            key = self.last_requests.pop(0)  # pop out first element
            del self.data_cache[key]  # delete this element from cache
            if self.verbose:
                print(f'Deleted chunk {key} from cache.')

            count += 1
            if count >= 100:
                # If something goes wrong just flush the whole memory.
                if self.verbose:
                    print(f'Tried to delete more than 100 elements. Flushing cache ...')
                self._flush_cache()
                break

    def _flush_cache(self):
        """
        Deletes all elements from cache
        """
        key_list = list(self.data_cache.keys())
        for k in key_list:
            del self.data_cache[key]
        self.last_requests = []

    def _pull_data(self, fi_ci, ind):
        """
        Retrieves the requested data from the cache.
        If the cache is empty at the requested address, load the data from
        the source h5-file and keep it in the cache.
        """
        slicer = self._transform_indices_to_slice(ind)

        # load chunk into cache
        if fi_ci not in self.data_cache.keys():
            fi = fi_ci[0]
            ci = fi_ci[1]
            self.data_cache[fi_ci] = dict()

            with h5py.File(self.h5_files[fi], 'r') as hf:
                for k in self.dataset_keys:
                    start = int(ci * self.chunk_size)
                    end = int((ci + 1) * self.chunk_size)
                    self.data_cache[fi_ci][k] = hf[k][start:end]
                if self.verbose:
                    print(f'Cached chunk {ci} ({start}:{end}) from file {fi}: {self.h5_files[fi]}.')

        # get data
        data = {k: self.data_cache[fi_ci][k][slicer] for k in self.dataset_keys}
        return data

    def _get_samples(self, i):
        """
        Converts the requested indices into cache addresses, calls the cache retriever
        function, updates the request history, and cleans the cache.
        """

        # get address
        idxs = self._transform_slice_to_indices(i)
        adrs = [self._idx2address(i) for i in idxs]
        fi_ci = [(a[0], a[1]) for a in adrs]  # file and chunk numbers
        li = [a[2] for a in adrs]  # local indices

        # get samples
        data_chunks = []
        pre = fi_ci[0]
        ind = []
        for i, a in enumerate(fi_ci):
            if a == pre:
                ind.append(li[i])
            else:
                data_chunks.append(self._pull_data(pre, ind))
                pre = a
                ind = [li[i]]

        data_chunks.append(self._pull_data(pre, ind))

        # save request address in history
        for adr in fi_ci:
            if adr not in set(self.last_requests):
                self.last_requests.append(adr)

        # clean cache
        self._clean_cache()

        return data_chunks

    def __getitem__(self, i):
        """
        Interface function for retrieving samples from the dataset.
        """
        # send the request
        data = self._get_samples(i)

        # reformat data from chunks to output format
        vol_data = np.concatenate([chunk['vol_data'] for chunk in data])
        vol_labels = np.concatenate([chunk['vol_labels'] for chunk in data])

        return vol_data, vol_labels