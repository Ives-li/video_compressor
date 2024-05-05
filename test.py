import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch 
from cacti_utils.utils import save_single_image,get_device_info,load_checkpoints
from cacti_utils.metrics import compare_psnr,compare_ssim
from cacti_utils.config import Config
from cacti_utils.logger import Logger
from torch.cuda.amp import autocast
import numpy as np 
import argparse 
import time
import einops 
from stformer import STFormer
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str)
    parser.add_argument("--weights",type=str)
    parser.add_argument("--device",type=str,default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device="cpu"
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    device = args.device
    config_name = osp.splitext(osp.basename(args.config))[0]
    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',config_name)
    if args.weights is None:
        args.weights = cfg.checkpoints

    test_dir = osp.join(args.work_dir,"reconstruct_images")

    log_dir = osp.join(args.work_dir,"test_log")
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)

    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])
    logger.info('GPU info:\n' 
            + dash_line + 
            env_info + '\n' +
            dash_line) 
    

    cr=8
    meas_dir = "work_dirs\\gray_data\\meas_images"
    meas_dir = os.path.abspath(os.path.join(project_root, meas_dir))

    all_meas = []
    meas_name_list = os.listdir(meas_dir)
    for file_name in meas_name_list:
        file_path = os.path.join(meas_dir, file_name)
        meas = Image.open(file_path)
        meas = np.array(meas, dtype=np.float32) 
        all_meas.append(meas)
    meas = np.array(all_meas)
    
    mask_dir = "work_dirs\\gray_data\\mask_images"
    mask_dir = os.path.abspath(os.path.join(project_root, mask_dir))

    all_masks = []
    mask_name_list = os.listdir(mask_dir)
    for file_name in mask_name_list:
        file_path = os.path.join(mask_dir, file_name)
        mask = Image.open(file_path)
        mask = np.array(mask, dtype=np.float32) 
        all_masks.append(mask)
    mask = np.array(all_masks)
    

    mask_s = np.sum(mask, axis=0)
    mask_s[mask_s == 0] = 1  # Avoid division by zero in later computations



    model = STFormer(color_channels=1,units=4,dim=64).to(device)
    logger.info("Load pre_train model...")
    resume_dict = torch.load(cfg.checkpoints)
    if "model_state_dict" not in resume_dict.keys():
        model_state_dict = resume_dict
    else:
        model_state_dict = resume_dict["model_state_dict"]
    load_checkpoints(model,model_state_dict)

    

    Phi = einops.repeat(mask,'cr h w->b cr h w',b=1)
    Phi_s = einops.repeat(mask_s,'h w->b 1 h w',b=1)
    Phi = torch.from_numpy(Phi).to(args.device)
    Phi_s = torch.from_numpy(Phi_s.astype(np.float32)).to(args.device)
    sum_time=0.0
    time_count = 0

    batch_output = []


    meas = torch.tensor(meas).float().to(device)
    batch_size = meas.shape[0]
    


    for ii in range(batch_size):
        single_meas = meas[ii].unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            outputs = model(single_meas, Phi, Phi_s)
            torch.cuda.synchronize()
            end = time.time()
            run_time = end - start
            if ii>0:
                sum_time += run_time
                time_count += 1
        if not isinstance(outputs,list):
            outputs = [outputs]
        output = outputs[-1][0].cpu().numpy().astype(np.float32)
        batch_output.append(output)

    #save image
    out = np.array(batch_output)
    for j in range(out.shape[0]):
        image_dir = osp.join(test_dir)
        if not osp.exists(image_dir):
            os.makedirs(image_dir)
        save_single_image(out[j],image_dir,j,name=config_name)
    if time_count==0:
        time_count=1
    logger.info('Average Run Time:\n' 
            + dash_line + 
            "{:.4f} s.".format(sum_time/time_count) + '\n' +
            dash_line)
    logger.info("Regenerate Successfully: \n")

if __name__=="__main__":
    main()