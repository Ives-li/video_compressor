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
import numpy as np 
import argparse 
import time
import einops 
from stformer import STFormer
from imageDataset import GrayData
import cv2

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("orig_folder",type=str)
    parser.add_argument("--work_dir",type=str)
    parser.add_argument("--weights",type=str)
    parser.add_argument("--device",type=str,default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device="cpu"
    return args

def load_model():
    device = "cuda:0"
    if not torch.cuda.is_available():
        device="cpu"   
    model = STFormer(color_channels=1,units=4,dim=64).to(device)
    log_relative = 'work_dirs\\gray_data\\test_log'
    log_dir = os.path.abspath(os.path.join(project_root, log_relative))
    logger = Logger(log_dir)
    logger.info("Load pre_train model...")
    checkpoints_relative = 'stformer_base.pth'
    checkpoin_path = os.path.abspath(os.path.join(project_root, checkpoints_relative))
    resume_dict = torch.load(checkpoin_path)
    if "model_state_dict" not in resume_dict.keys():
        model_state_dict = resume_dict
    else:
        model_state_dict = resume_dict["model_state_dict"]
    load_checkpoints(model,model_state_dict)   

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    orig_folder = args.orig_folder

    device = args.device
    config_name = osp.splitext(osp.basename(args.config))[0]
    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',config_name)
    if args.weights is None:
        args.weights = cfg.checkpoints

    current_directory = os.path.dirname(current_file_path)
    result_dir = os.path.join(current_directory, "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    test_dir = osp.join(result_dir,"test_images")



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
    
    # data_root_relative = 'frames\\orig_frames'
    # data_root = os.path.abspath(os.path.join(project_root, data_root_relative))
    data_root = orig_folder

    cr=8
    test_data = GrayData(data_root, cr, frame_height=128, frame_width=128)
    meas, gt, mask = test_data.process_all_data() #gt for evaluation

    mask_s = np.sum(mask, axis=0)
    mask_s[mask_s == 0] = 1  # Avoid division by zero in later computations

    meas_pre_dir = "meas_pre_images"
    meas_dir = "meas_images"
    meas_pre_dir = os.path.abspath(os.path.join(result_dir, meas_pre_dir))
    meas_dir = os.path.abspath(os.path.join(result_dir, meas_dir))
    if not osp.exists(meas_pre_dir):
        os.makedirs(meas_pre_dir)
    if not osp.exists(meas_dir):
        os.makedirs(meas_dir)
    save_single_image(meas, meas_pre_dir, 0)
    for i in range(meas.shape[0]):
        cv2.imwrite(osp.join(meas_dir,str(i+1)+".png"),meas[i])

    mask_pre_dir = "mask_pre_images"
    mask_dir = "mask_images"
    mask_pre_dir = os.path.abspath(os.path.join(result_dir, mask_pre_dir))
    mask_dir = os.path.abspath(os.path.join(result_dir, mask_dir))
    if not osp.exists(mask_pre_dir):
        os.makedirs(mask_pre_dir)
    if not osp.exists(mask_dir):
        os.makedirs(mask_dir)
    save_single_image(mask, mask_pre_dir, 0)
    for i in range(mask.shape[0]):
        cv2.imwrite(osp.join(mask_dir,str(i+1)+".png"),mask[i])

    model = STFormer(color_channels=1,units=4,dim=64,frames=(mask.shape[0]*meas.shape[0])).to(device)
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
    Phi_s = torch.from_numpy(Phi_s).to(args.device)
        
    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    sum_time=0.0
    time_count = 0
    
    psnr,ssim = 0,0
    batch_output = []


    meas = torch.tensor(meas).float().to(device)
    batch_size = meas.shape[0]
    


    out_list = []
    gt_list = []
    for ii in range(batch_size):
        single_gt = gt[ii]
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
        
        for jj in range(cr):
            # if output.shape[0]==3:
            #     per_frame_out = output[:,jj]
            #     rgb2raw = test_data.rgb2raw
            #     per_frame_out = np.sum(per_frame_out*rgb2raw,axis=0)
            # else:
            per_frame_out = output[jj]
            per_frame_gt = single_gt[jj]

            psnr += compare_psnr(per_frame_gt*255,per_frame_out*255)
            ssim += compare_ssim(per_frame_gt*255,per_frame_out*255)
            
    meas_num = len(batch_output)
    psnr = psnr / (meas_num* cr)
    ssim = ssim / (meas_num* cr)
    logger.info(" Mean PSNR: {:.4f} Mean SSIM: {:.4f}.".format(psnr,ssim))
    psnr_list.append(psnr)
    ssim_list.append(ssim)

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

    psnr_dict["psnr_mean"] = np.mean(psnr_list)
    ssim_dict["ssim_mean"] = np.mean(ssim_list)

    psnr_str = ", ".join([key+": "+"{:.4f}".format(psnr_dict[key]) for key in psnr_dict.keys()])
    ssim_str = ", ".join([key+": "+"{:.4f}".format(ssim_dict[key]) for key in ssim_dict.keys()])
    logger.info("Mean PSNR: \n"+
                dash_line + 
                "{}.\n".format(psnr_str)+
                dash_line)

    logger.info("Mean SSIM: \n"+
                dash_line + 
                "{}.\n".format(ssim_str)+
                dash_line) 

if __name__=="__main__":
    main()