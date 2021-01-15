from VISoR_Brain.format.raw_data import RawData
from VISoR_Brain.positioning.visor_sample import get_column_images
import torch
import numpy as np
from torch.utils.data import Dataset

class ColumnDataset(Dataset):
    def __init__(self, data_file, roi_list, source='raw'):
        self.raw_data = RawData(data_file)
        self.roi_list = roi_list
        self.source = source
        self.count = 0
        self.prev_col = None
        self.prev_end = None
        for r in self.roi_list:
            if r is not None:
                self.prev_col = r[0]
                self.prev_end = r[1][2]
                break
        if self.prev_col is None:
            raise AssertionError('No data needs to load.')

    def __getitem__(self, item):
        if self.roi_list[item] is None:
            return torch.zeros((10, 10, 10))
        c, r0, r1 = self.roi_list[item]
        r0, r1 = np.minimum(r0, r1), np.maximum(r0, r1)
        r0, r1 = np.int32(np.floor((np.maximum(r0, 0)))), np.int32(np.ceil(np.minimum(r1, self.raw_data.size[3])))
        s, e = r0[2], r1[2]
        img = self.raw_data.load(c, [s, e], source_type=self.source, output_format='numpy')
        self.count += self.prev_end
        if c != self.prev_col or self.count > 3000:
            self.prev_col = c
            self.prev_end = s
            self.raw_data.release()
        else:
            self.count += e - self.prev_end
        img = torch.Tensor(img.view(np.float32))
        return img

    def __len__(self):
        return len(self.roi_list)
