import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from cacti_utils.utils import save_single_image

class GrayData(Dataset):
    def __init__(self, data_root, frames_per_measurement, frame_height, frame_width, *args, **kwargs):
        self.data_root = data_root
        self.data_name_list = os.listdir(data_root)
        self.frames_per_measurement = frames_per_measurement
        self.frame_height = frame_height
        self.frame_width = frame_width

    def process_all_data(self):
        # Storage for all data
        all_pics = []

        # Process each image in the directory
        for file_name in self.data_name_list:
            file_path = os.path.join(self.data_root, file_name)
            pic = Image.open(file_path)
            pic = np.array(pic, dtype=np.float32) / 255.0  # Normalize
            all_pics.append(pic)
        
        pics = np.array(all_pics)
        pics = pics[:, :, :, 0]
        pics = np.transpose(pics, (1, 2, 0))
        pics = pics[0:self.frame_height,0:self.frame_width,:]
        frames = pics.shape[2] // self.frames_per_measurement
        pic_gt = np.zeros([self.frames_per_measurement, frames, self.frame_height, self.frame_width])

        mask_shape = (frames, self.frame_height, self.frame_width)
        mask = np.random.randint(0, 2, size=mask_shape)
        mask = mask.astype(np.float32)  # Ensure mask is float for further operations



        for jj in range(pics.shape[2]):
            if jj % frames == 0:
                meas_t = np.zeros([self.frame_height, self.frame_width])
                n = 0
            pic_t = pics[:, :, jj]
            mask_t = mask[n, :, :]

            pic_gt[jj // frames, n, :, :] = pic_t
            n += 1
            meas_t = meas_t + np.multiply(mask_t, pic_t)

            if jj == ((pics.shape[2] // self.frames_per_measurement)-1):
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % (pics.shape[2] // self.frames_per_measurement) == 0 and jj != ((pics.shape[2] // self.frames_per_measurement)-1):
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        return meas,pic_gt, mask

    def __len__(self):
        return len(self.data_name_list)
    
