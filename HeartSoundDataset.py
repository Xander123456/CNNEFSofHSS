import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset

class HeartSoundDataset(Dataset):

    def __init__(self, data_dir, label_csv, length=0, sample_rate=32000):
        super(HeartSoundDataset, self).__init__()

        self.length = length
        self.sample_rate = sample_rate
        self.data_list = []

        with open(label_csv, "r") as csv_f:
            csv_reader = csv.reader(csv_f, delimiter=',')

            for row in csv_reader:
                file_name = row[0]
                file_label = [1.0, 0.0] if int(row[1]) == -1 else [0.0, 1.0]

                np_data = np.load(os.path.join(data_dir, f"{file_name}.npy"))

                if length == 0:
                    self.data_list.append([torch.Tensor(np_data), torch.Tensor(file_label)])

                else:
                    x = int(np_data.shape[0] / (self.sample_rate * self.length))
                    for i in range(x):

                        d = np_data[i * self.sample_rate * self.length : (i + 1) * self.sample_rate * self.length]

                        self.data_list.append([torch.Tensor(d), torch.Tensor(file_label)])

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":
    h = HeartSoundDataset("dataset", "label/PCCD/0/a_train.csv")
    print(len(h))

