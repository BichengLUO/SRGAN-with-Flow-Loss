#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import config, log_config
from flow import *

def evaluate(tag, is_no_video):
    if is_no_video:
        save_dir = "samples_{}/evaluate".format(tag)
    else:
        save_dir = "samples_video_{}/evaluate".format(tag)
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    gen_img_list = sorted(tl.files.load_file_list(path=save_dir, regx='valid_gen.*.png', printable=False))
    bicu_img_list = sorted(tl.files.load_file_list(path=save_dir, regx='valid_bicubic.*.png', printable=False))
    
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    gen_imgs = tl.vis.read_images(gen_img_list, path=save_dir, n_threads=32)
    bicu_imgs = tl.vis.read_images(bicu_img_list, path=save_dir, n_threads=32)
    
    t_valid_hr_img = tf.placeholder('float32', [1, None, None, 3], name='hr_image')
    t_gen_img = tf.placeholder('float32', [1, None, None, 3], name='gen_image')
    t_bicu_img = tf.placeholder('float32', [1, None, None, 3], name='bicu_image')
    
    t_valid_hr_flow = tf.placeholder('float32', [1, None, None, 2], name='hr_flow')
    t_gen_flow = tf.placeholder('float32', [1, None, None, 2], name='gen_flow')
    t_bicu_flow = tf.placeholder('float32', [1, None, None, 2], name='bicu_flow')
    
    gen_psnr = tf.image.psnr(t_gen_img, t_valid_hr_img, max_val=255)
    bicu_psnr = tf.image.psnr(t_bicu_img, t_valid_hr_img, max_val=255)
    gen_ssim = tf.image.ssim(t_gen_img, t_valid_hr_img, max_val=255)
    bicu_ssim = tf.image.ssim(t_bicu_img, t_valid_hr_img, max_val=255)
    
    gen_flow_mse = tl.cost.mean_squared_error(t_gen_flow, t_valid_hr_flow, is_mean=True)
    bicu_flow_mse = tl.cost.mean_squared_error(t_bicu_flow, t_valid_hr_flow, is_mean=True)
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    
    f = open(save_dir + "/psnr_ssim.csv", "w")
    f.write('Frame, gen_psnr, bicu_psnr, gen_ssim, bicu_ssim\n')
    for idx in range(len(valid_hr_imgs)):
        valid_hr_img = valid_hr_imgs[idx]
        gen_img = gen_imgs[idx]
        bicu_img = bicu_imgs[idx]
        genPsnr, bicuPsnr, genSsim, bicuSsim = sess.run([gen_psnr, bicu_psnr, gen_ssim, bicu_ssim], {t_valid_hr_img: [valid_hr_img], t_gen_img: [gen_img], t_bicu_img: [bicu_img]})
        print('Frame [%04d]: gen_psnr: %.8f, bicu_psnr: %.8f, gen_ssim: %.8f, bicu_ssim: %.8f' % (idx+1, genPsnr, bicuPsnr, genSsim, bicuSsim))
        f.write('%04d, %.8f, %.8f, %.8f, %.8f\n' % (idx+1, genPsnr, bicuPsnr, genSsim, bicuSsim))
       
    f = open(save_dir + "/flow_mse.csv", "w")
    f.write('Frame, gen_flow_mse, bicu_flow_mse\n')
    for idx in range(len(valid_hr_imgs)-1):
        valid_hr_2_frames = np.stack(valid_hr_imgs[idx:idx+2])
        gen_2_frames = np.stack(gen_imgs[idx:idx+2])
        bicu_2_frames = np.stack(bicu_imgs[idx:idx+2])
        
        valid_hr_flow = gen_flows(valid_hr_2_frames, is_transformed=False)
        gen_flow = gen_flows(gen_2_frames, is_transformed=False)
        bicu_flow = gen_flows(bicu_2_frames, is_transformed=False)
        
        genFlowMse, bicuFlowMse = sess.run([gen_flow_mse, bicu_flow_mse], {t_valid_hr_flow: valid_hr_flow, t_gen_flow: gen_flow, t_bicu_flow: bicu_flow})
        print('Frame [%04d]: gen_flow_mse: %.8f, bicu_flow_mse: %.8f' % (idx+1, genFlowMse, bicuFlowMse))
        f.write('%04d, %.8f, %.8f\n' % (idx+1, genFlowMse, bicuFlowMse))
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='', help='A tag for net outputs')
    parser.add_argument('--no_video', action='store_true', default=False, help='Not operating on a video folder')

    args = parser.parse_args()
    evaluate(args.tag, args.no_video)