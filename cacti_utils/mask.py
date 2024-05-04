import numpy as np 
import scipy.io as scio 

def generate_masks(mask_path=None,mask_shape=None):
    assert mask_path is not None or mask_shape is not None
    if mask_path is None:
        mask = np.random.randint(0,2,size=(mask_shape[0],mask_shape[1],mask_shape[2]))
        # masks = [np.random.randint(0, 2, (frame_height, frame_width)) for _ in range(frames_per_measurement)]

    else:
        mask = scio.loadmat(mask_path)
        mask = mask['mask']
        if mask_shape is not None:
            h,w,c = mask.shape
            m_h,m_w,m_c = mask_shape[0],mask_shape[1],mask_shape[2]
            h_b = np.random.randint(0,h-m_h+1)
            w_b = np.random.randint(0,w-m_w+1)
            mask = mask[h_b:h_b+m_h,w_b:w_b+m_w,:m_c]
    mask = np.transpose(mask, [2, 0, 1])
    mask = mask.astype(np.float32)

    mask_s = np.sum(mask, axis=0)
    mask_s[mask_s==0] = 1
    return mask, mask_s