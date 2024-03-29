from os.path import join
import pandas as pd
from .base_dataset import BaseDataset


class retail786k_256Dataset(BaseDataset):

    def __init__(self, data_dir, mode, transform=None, load_super_labels=False, **kwargs):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        if mode == 'train':
            mode = ['train']
        elif mode == 'test':
            mode = ['test']
        elif mode == 'all':
            mode = ['train', 'test']
        else:
            raise ValueError(f"Mode unrecognized {mode}")

        self.paths = []
        self.labels = []
        self.super_labels = []
        for splt in mode:
            gt = pd.read_csv(join(self.data_dir, f'retail-786k_256_info_all_{splt}.txt'), sep=' ')
            self.paths.extend(gt["path"].apply(lambda x: join(self.data_dir, x)).tolist())
            self.labels.extend((gt["class_id"]).tolist())
            self.super_labels.extend((gt["super_class_id"]).tolist())

        self.get_instance_dict()
        self.get_super_dict()
