"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep
import matplotlib.pyplot as plt
import cv2

def main(args):
    sleep(random.random())
        
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
    minsize = 40 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    #threshold = [ 0.9, 0.9, 0.9 ]  # three steps's threshold
    factor = 0.709 # scale factor

    image_path = args.input_image
    print(image_path)

    img = misc.imread(image_path)
    if img.ndim<2:
        print('Unable to align "%s"' % image_path)
        return
    if img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:,:,0:3]

    bounding_boxes, points = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    print('No. face: '+str(nrof_faces))
                    
    if nrof_faces>0:
        plt.figure()
        plt.imshow(img)
        for faceI in range(nrof_faces):
            plt.gca().add_patch(plt.Rectangle(bounding_boxes[faceI][0:2], 
                   bounding_boxes[faceI][2]-bounding_boxes[faceI][0], bounding_boxes[faceI][3]-bounding_boxes[faceI][1], 
                   fill=False, edgecolor='r', linewidth=3))
            plt.scatter(points[0:5,faceI], points[5:,faceI])
            plt.text(bounding_boxes[faceI][2], bounding_boxes[faceI][3], str(bounding_boxes[faceI][4]), color='red')
        plt.show()
        
        all_det = bounding_boxes[:,0:4]

        #--- The one has the largest box size and the closest to the image center
#        img_size = np.asarray(img.shape)[0:2]
#        bounding_box_size = (all_det[:,2]-all_det[:,0])*(all_det[:,3]-all_det[:,1])
#        img_center = img_size / 2
#        offsets = np.vstack([ (all_det[:,0]+all_det[:,2])/2-img_center[1], (all_det[:,1]+all_det[:,3])/2-img_center[0] ])
#        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
#        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
        
        #--- The one has the largest probability of being a face
        #index = np.argmax(bounding_boxes[:,4])  
        
        #--- The one has the largest box size and the largest probability of being a face
        bounding_box_size = (all_det[:,2]-all_det[:,0])*(all_det[:,3]-all_det[:,1])
        index = np.argmax(bounding_box_size*np.power(bounding_boxes[:,4]-0.98,3.0))
        print(bounding_box_size*np.power(bounding_boxes[:,4]-0.98,3.0))
               
        det = all_det[index,:]
        det = np.squeeze(det)
        
        wFace = det[2]-det[0]
        hFace = det[3]-det[1]
        imageCenter = ((det[2]+det[0])//2, (det[3]+det[1])//2) 
        pointRatioW = (points[:5, index] - imageCenter[0])/wFace
        pointRatioH = (points[5:, index] - imageCenter[1])/hFace
        print(pointRatioW)
        print(pointRatioH)
            
        leftEye = (points[0, index], points[5, index])
        rightEye = (points[1, index], points[6, index])
        eyesCenter = ((leftEye[0]+rightEye[0])//2, (leftEye[1]+rightEye[1])//2) 
        dY = rightEye[1] - leftEye[1]
        dX = rightEye[0] - leftEye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        eyeDist = np.sqrt((dX**2)+(dY**2))
        scale = 1       # make the affine transformation with scale 1
        
        raw_image_size = int(args.image_size*eyeDist/args.eye_dist)
        raw_eye_margin = int(args.eye_margin*eyeDist/args.eye_dist)

        # grab the rotation matrix for rotating and scaling the face
        transM = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        transM[0, 2] += (raw_image_size//2 - eyesCenter[0])
        transM[1, 2] += (raw_eye_margin - eyesCenter[1])
        
        cropped = cv2.warpAffine(img, transM, (raw_image_size, raw_image_size), flags=cv2.INTER_CUBIC)
        plt.figure()
        plt.imshow(cropped)
        plt.show()
        
        rescaled = cv2.resize(cropped, (args.image_size, args.image_size), interpolation=cv2.INTER_CUBIC)
        plt.figure()
        plt.imshow(rescaled)
        plt.show()
    else:
        print('Unable to align "%s"' % image_path)
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_image', type=str, help='Directory with unaligned images.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--eye_dist', type=int,
        help='Pixels between the two eyes.', default=70)
    parser.add_argument('--eye_margin', type=int,
        help='The distance between the eye and the upper boundary.', default=50)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
