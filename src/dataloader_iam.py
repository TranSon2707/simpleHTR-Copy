# import pickle
# import random
# from collections import namedtuple
# from typing import Tuple

# import cv2
# import lmdb
# import numpy as np
# from path import Path

# Sample = namedtuple('Sample', 'gt_text, file_path')
# Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size')


# class DataLoaderIAM:
#     """
#     Loads data which corresponds to IAM format,
#     see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
#     """

#     def __init__(self,
#                  data_dir: Path,
#                  batch_size: int,
#                  train_split: float = 0.7,
#                  val_split: float = 0.15,
#                  # test_split will be 1 - train_split - val_split = 0.15
#                  fast: bool = True) -> None:
#         """Loader for dataset."""

#         assert data_dir.exists()

#         self.fast = fast
#         if fast:
#             self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

#         self.data_augmentation = False
#         self.curr_idx = 0
#         self.batch_size = batch_size
#         self.samples = []

#         f = open(data_dir / 'gt/words.txt')
#         chars = set()
#         bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset
#         for line in f:
#             # ignore empty and comment lines
#             line = line.strip()
#             if not line or line[0] == '#':
#                 continue

#             line_split = line.split(' ')
#             assert len(line_split) >= 9

#             # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
#             file_name_split = line_split[0].split('-')
#             file_name_subdir1 = file_name_split[0]
#             file_name_subdir2 = f'{file_name_split[0]}-{file_name_split[1]}'
#             file_base_name = line_split[0] + '.png'
#             file_name = data_dir / 'img' / file_name_subdir1 / file_name_subdir2 / file_base_name

#             if line_split[0] in bad_samples_reference:
#                 print('Ignoring known broken image:', file_name)
#                 continue

#             # GT text are columns starting at 9
#             gt_text = ' '.join(line_split[8:])
#             chars = chars.union(set(list(gt_text)))

#             # put sample into list
#             self.samples.append(Sample(gt_text, file_name))

#         # Split into training, validation and test sets
#         train_idx = int(train_split * len(self.samples))
#         val_idx = train_idx + int(val_split * len(self.samples))
        
#         self.train_samples = self.samples[:train_idx]   # training set
#         self.validation_samples = self.samples[train_idx:val_idx]   # validation set
#         self.test_samples = self.samples[val_idx:]  # test set

#         # Put words into lists
#         self.train_words = [x.gt_text for x in self.train_samples]
#         self.validation_words = [x.gt_text for x in self.validation_samples]
#         self.test_words = [x.gt_text for x in self.test_samples]  # New test words

#         # start with train set
#         self.train_set()

#         # list of all chars in dataset
#         self.char_list = sorted(list(chars))

#     def train_set(self) -> None:
#         """Switch to randomly chosen subset of training set."""
#         self.data_augmentation = True
#         self.curr_idx = 0
#         random.shuffle(self.train_samples)
#         self.samples = self.train_samples
#         self.curr_set = 'train'

#     def validation_set(self) -> None:
#         """Switch to validation set."""
#         self.data_augmentation = False
#         self.curr_idx = 0
#         self.samples = self.validation_samples
#         self.curr_set = 'val'

#     def test_set(self) -> None:
#         """Switch to test set."""
#         self.data_augmentation = False
#         self.curr_idx = 0
#         self.samples = self.test_samples
#         self.curr_set = 'test'

#     def get_iterator_info(self) -> Tuple[int, int]:
#         """Current batch index and overall number of batches."""
#         if self.curr_set == 'train':
#             num_batches = int(np.floor(len(self.samples) / self.batch_size))  # train set: only full-sized batches
#         else:
#             num_batches = int(np.ceil(len(self.samples) / self.batch_size))  # val/test set: allow last batch to be smaller
#         curr_batch = self.curr_idx // self.batch_size + 1
#         return curr_batch, num_batches

#     def has_next(self) -> bool:
#         """Is there a next element?"""
#         if self.curr_set == 'train':
#             return self.curr_idx + self.batch_size <= len(self.samples)  # train set: only full-sized batches
#         else:
#             return self.curr_idx < len(self.samples)  # val/test set: allow last batch to be smaller

#     def _get_img(self, i: int) -> np.ndarray:
#         if self.fast:
#             with self.env.begin() as txn:
#                 basename = Path(self.samples[i].file_path).basename()
#                 data = txn.get(basename.encode("ascii"))
#                 img = pickle.loads(data)
#         else:
#             img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)

#         return img

#     def get_next(self) -> Batch:
#         """Get next element."""
#         batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

#         imgs = [self._get_img(i) for i in batch_range]
#         gt_texts = [self.samples[i].gt_text for i in batch_range]

#         self.curr_idx += self.batch_size
#         return Batch(imgs, gt_texts, len(imgs))


import pickle
import random
from collections import namedtuple
from typing import Tuple

import cv2
import lmdb
import numpy as np
from path import Path

Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size')


class DataLoaderIAM:
    """
    Loads data which corresponds to IAM format,
    see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    def __init__(self,
                 data_dir: Path,
                 batch_size: int,
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 fast: bool = True) -> None:
        """Loader for dataset."""

        assert data_dir.exists()
        assert train_split + val_split < 1.0, "Train + Val must be < 1.0"

        self.fast = fast
        if fast:
            self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []

        # ------------------------------------------------------------------
        # Read GT
        # ------------------------------------------------------------------
        f = open(data_dir / 'gt/words.txt')
        chars = set()
        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']

        for line in f:
            line = line.strip()
            if not line or line[0] == '#':
                continue

            line_split = line.split(' ')
            assert len(line_split) >= 9

            file_name_split = line_split[0].split('-')
            file_name_subdir1 = file_name_split[0]
            file_name_subdir2 = f'{file_name_split[0]}-{file_name_split[1]}'
            file_base_name = line_split[0] + '.png'
            file_name = data_dir / 'img' / file_name_subdir1 / file_name_subdir2 / file_base_name

            if line_split[0] in bad_samples_reference:
                print('Ignoring known broken image:', file_name)
                continue

            gt_text = ' '.join(line_split[8:])
            chars = chars.union(set(gt_text))

            self.samples.append(Sample(gt_text, file_name))

        # ------------------------------------------------------------------
        # Split dataset: 70% train - 15% val - 15% test
        # ------------------------------------------------------------------
        random.seed(42)
        random.shuffle(self.samples)

        num_samples = len(self.samples)
        train_end = int(train_split * num_samples)
        val_end = int((train_split + val_split) * num_samples)

        self.train_samples = self.samples[:train_end]
        self.validation_samples = self.samples[train_end:val_end]
        self.test_samples = self.samples[val_end:]

        # word lists
        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]
        self.test_words = [x.gt_text for x in self.test_samples]

        print(f"Train samples: {len(self.train_samples)}")
        print(f"Validation samples: {len(self.validation_samples)}")
        print(f"Test samples: {len(self.test_samples)}")

        # start with train set
        self.train_set()

        # list of all chars in dataset
        self.char_list = sorted(list(chars))

    # ------------------------------------------------------------------
    # Dataset switching
    # ------------------------------------------------------------------
    def train_set(self) -> None:
        """Switch to training set."""
        self.data_augmentation = True
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self) -> None:
        """Switch to validation set."""
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def test_set(self) -> None:
        """Switch to test set."""
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.test_samples
        self.curr_set = 'test'

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------
    def get_iterator_info(self) -> Tuple[int, int]:
        """Current batch index and overall number of batches."""
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))

        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self) -> bool:
        """Is there a next batch?"""
        if self.curr_set == 'train':
            return self.curr_idx + self.batch_size <= len(self.samples)
        else:
            return self.curr_idx < len(self.samples)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _get_img(self, i: int) -> np.ndarray:
        if self.fast:
            with self.env.begin() as txn:
                basename = Path(self.samples[i].file_path).basename()
                data = txn.get(basename.encode("ascii"))
                img = pickle.loads(data)
        else:
            img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)

        return img

    def get_next(self) -> Batch:
        """Get next batch."""
        batch_range = range(
            self.curr_idx,
            min(self.curr_idx + self.batch_size, len(self.samples))
        )

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs))
