import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class ImageData(Dataset):
    def __init__(self, data_root, frame_height, frame_width, frames_per_measurement, rot_flip_flag=False, transpose=False, partition=None):
        self.data_root = data_root
        self.data_name_list = os.listdir(data_root)
        self.rot_flip_flag = rot_flip_flag
        self.transpose = transpose
        self.partition = partition
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frames_per_measurement = frames_per_measurement
        self.mask, self.mask_s = self.generate_mask()

    def generate_mask(self):
        mask = np.random.randint(0, 2, size=(self.frames_per_measurement, self.frame_height, self.frame_width))
        mask_s = np.sum(mask, axis=0)
        mask_s[mask_s == 0] = 1
        return mask, mask_s
    
    def get_mask(self):
        return self.mask, self.mask_s

    def __getitem__(self, index):
        file_path = os.path.join(self.data_root, self.data_name_list[index])
        pic = Image.open(file_path)
        pic = np.array(pic) / 255.0
        
        if pic.ndim == 2:  # Grayscale to RGB if needed
            pic = np.stack([pic]*3, axis=-1)

        if self.transpose:
            pic = pic.transpose(2, 0, 1)  # C, H, W

        measurements = []
        pic_gt = np.zeros((self.frames_per_measurement, 3, self.frame_height, self.frame_width), dtype=np.float32)

        for jj in range(self.frames_per_measurement):
            mask_t = self.mask[jj]
            meas_t = mask_t * pic
            measurements.append(meas_t)
            pic_gt[jj] = pic

        measurements = np.array(measurements)
        
        if self.partition is not None:
            img_h, img_w = self.frame_height, self.frame_width
            part_h, part_w = self.partition['height'], self.partition['width']
            assert (img_h % part_h == 0) and (img_w % part_w == 0), "Image cannot be chunked!"
            h_num = img_h // part_h
            w_num = img_w // part_w
            measurements = measurements.reshape(-1, part_h, part_w)
            pic_gt = pic_gt.reshape(-1, 3, part_h, part_w)

        return measurements, pic_gt

    def __len__(self):
        return len(self.data_name_list)
