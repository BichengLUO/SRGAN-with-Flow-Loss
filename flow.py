#! /usr/bin/python
# -*- coding: utf8 -*-

import cv2
import numpy as np
import tensorflow as tf
import tensorlayer as tl

# Generates optical flows between each frame of the images batch
# Parameters:
# -------------
#     imgs: Numpy arrays of the images batch
#     is_transformed: Indicates whether the image has been transformed to [-1, 1] instead of [0, 255]
#
# Returns:
# -------------
#     Returns numpy arrays of the optical flows
def gen_flows(imgs, is_transformed=True):
    if is_transformed:
        imgs = ((imgs + 1.) * 255. / 2.).astype(np.uint8)
    batch_size = imgs.shape[0]
    prv = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
    flows = []
    for i in range(1, batch_size):
        nxt = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prv, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
        prv = nxt
    return np.stack(flows)

# Color encodes the optical flows for visualization
# Parameters:
# -------------
#     imgs: Numpy arrays of the optical flows
#
# Returns:
# -------------
#     Returns numpy arrays of the optical flow visualization images
def vis_flows(flows):
    batch_size = flows.shape[0]
    flow_vis = []
    for i in range(batch_size):
        flow = flows[i]
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,1] = 255
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        bgr = bgr.astype(np.float32) / (255. / 2.)
        bgr = bgr - 1.
        flow_vis.append(bgr)
    return np.stack(flow_vis)

# Merges batch of images with batch of flows for gradient descent
# Parameters:
# -------------
#     imgs: Tensor of the batch of images, i.e. frames, shape = [batch_size, 384, 384, 3]
#     flows: Tensor of the batch of flows, shape = [batch_size-1, 384, 384, 2]
#     alpha: Weight for the optical flow loss
#
# Returns:
# -------------
#     Returns ops to merge batch of images with batch of flows
def merge_imgs_flows(imgs, flows, alpha):
    batch_size = imgs.shape[0]
    # Remove one frame to align with the number of flows
    imgs = imgs[:batch_size-1]
    # Pads the flows to align with the number of image channels
    padding = tf.zeros([flows.shape[0], flows.shape[1], flows.shape[2], 1])
    merged = imgs + alpha * tf.concat([flows, padding], axis=3)
    return merged
    