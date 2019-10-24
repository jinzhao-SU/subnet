from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import torch


class UAVDatasetTuple(Dataset):
    def __init__(self, task_path, label_path):
        self.task_path = task_path
        self.label_path = label_path
        self.label_md = []
        self.task_md = []
        self._get_tuple()

    def __len__(self):
        return len(self.label_md)

    def _get_tuple(self):
        self.task_md = np.load(self.task_path).astype(float)
        self.label_md = np.load(self.label_path).astype(float)
        assert len(self.task_md) == len(self.label_md), "not identical"

    def __getitem__(self, idx):
        sample = {}
        try:
            task = self._prepare_task(idx)
            label = self._get_label(idx)
        except Exception as e:
            print('error encountered while loading {}'.format(idx))
            print("Unexpected error:", sys.exc_info()[0])
            print(e)
            raise

        sample = {'task': task, 'label': label}

        return sample

    def _prepare_task(self, idx):
        task_coordinate = self.task_md[idx]
        # print(task_coordinate[0])
        task_md = np.zeros((15, 100, 100))
        count = 0
        for i in range(15):
            x1 = int(task_coordinate[i][0])
            y1 = int(task_coordinate[i][1])
            x2 = int(task_coordinate[i][2])
            y2 = int(task_coordinate[i][3])
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                continue
            else:
                task_md[count][x1][y1] = 1.00
                task_md[count][x2][y2] = 1.00
                count += 1
        # print("positive count", count)
        return task_md.reshape(count,10000)

    def _get_label(self, idx):
        label_md = self.label_md[idx].reshape(100,100)
        return label_md

    def get_class_count(self):
        total = len(self.label_md) * self.label_md[0].shape[0] * self.label_md[0].shape[1]
        positive_class = 0
        for label in self.label_md:
            positive_class += np.sum(label)
        print("The number of positive image pair is:", positive_class)
        print("The number of negative image pair is:", total - positive_class)
        positive_ratio = positive_class / total
        negative_ratio = (total - positive_class) / total

        return positive_ratio, negative_ratio

if __name__ == '__main__':
    data_path ='/data/zzhao/uav_regression/feature_extraction_data/data_tasks.npy'
    label_path = '/data/zzhao/uav_regression/feature_extraction_data/label_density.npy'

    all_dataset = UAVDatasetTuple(task_path=data_path, label_path=label_path)
    sample = all_dataset[0]
    print(sample['task'].shape)
    count = 0

    for idx, val in enumerate(sample['task'][0]):
        if val == 1.00:
            print(idx)
    print(count)