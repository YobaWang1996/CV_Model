from glob import glob
from os.path import join, basename, isdir
from typing import List

from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

import numpy as np
import torch
import cv2
from torchvision.transforms import transforms


# to tensor3D
class ToFloatTensor3D(object):
    """ Convert videos to FloatTensors """
    def __init__(self, normalize=True):
        self._normalize = normalize

    def __call__(self, sample):
        X = sample
        # swap color axis because
        # numpy image: T x H x W x C
        X = X.transpose(0, 3, 1, 2)

        if self._normalize:
            X = X / 255.

        X = np.float32(X)
        return torch.from_numpy(X)


# dataset for video
class UCSDPed2(Dataset):
    def __init__(self, path, train_test, number_frame, step=1):
        super(UCSDPed2, self).__init__()

        self.path = path
        self.t = number_frame
        self.step = step

        # Train or Test directory
        if train_test == 'Train':
            self.dataset_dir = join(self.path, 'Train')
        elif train_test == 'Test':
            self.dataset_dir = join(self.path, 'Test')
        elif train_test == "Validate":
            self.dataset_dir = join(self.path, 'Validate')
        else:
            print("Wrong directory!")

        # Transform
        self.transform = ToFloatTensor3D()

        # Load all videos ids
        self.test_ids = self.load_videos_ids()

        # Other utilities
        self.cur_len = 0
        self.cur_video_id = None
        self.cur_video_frames = None
        self.cur_video_labels = None

    def load_videos_ids(self):
        # type: () -> List[str]
        """
        Loads the set of all videos ids.

        :return: The list of videos ids.
        """
        return sorted([basename(d) for d in glob(join(self.dataset_dir, '**')) if isdir(d)])

    def train_or_test(self, video_id):
        # type: (str) -> None
        """
        Sets the dataset in mode.

        :param video_id: the id of the video to train or test.
        """

        self.cur_video_frames, self.cur_video_labels = self.load_sequence_frames_labels(video_id)

        self.cur_len = int((len(self.cur_video_frames)-self.t) / self.step + 1)

    def load_sequence_frames_labels(self, video_id):
        """
        Loads a test video in memory.

        :param video_id: the id of the test video to be loaded
        :return: the video in a np.ndarray, with shape (n_frames, h, w, c).
        """

        cur_dir = join(self.dataset_dir, video_id)
        cur_file = join(cur_dir, video_id+'.txt')
        with open(cur_file, "r") as f:
            img_path_label = []
            for i in f:
                i = i.strip("\n")
                i = i.rstrip("\n")
                words = i.split()
                img_path_label.append((words[0], words[1]))

        cur_video_clip = []
        cur_video_label = []
        for img_path, label in img_path_label:
            img = cv2.imread(img_path)
            img = resize(img, output_shape=(224, 224), preserve_range=True)
            cur_video_clip.append(img)
            cur_video_label.append(int(label))
        cur_video_clip = np.stack(cur_video_clip)
        cur_video_label = np.stack(cur_video_label)
        return cur_video_clip, cur_video_label

    def videos_ids(self):
        # type: () -> List[str]
        """
        Returns all available test videos.
        """
        return self.test_ids

    def __getitem__(self, item):
        # # type: (int) -> tuple[torch.Tensor,int]
        """
        Provides the i-th example.
        """

        t = self.t
        step = self.step
        item = item * step

        clip = self.cur_video_frames[item:item + t]
        sample = self.transform(clip)
        label = self.cur_video_labels[item:item + t]

        return sample, label

    def __len__(self):
        """
        Returns the number of examples.
        """
        return self.cur_len

