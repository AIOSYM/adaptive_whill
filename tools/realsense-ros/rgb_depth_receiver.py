#!/usr/bin/env python
import cv2 
import rospy
import message_filters
import numpy as np

from sensor_msgs.msg import CompressedImage

frameID = 0

def showImage(rgb_image, depth_image):
    print('RGB shape: {}'.format(rgb_image.shape))
    print('Depth shape: {}'.format(depth_image.shape))

    cv2.imshow('RGB Image', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Depth Image', depth_image)
    
    print("Depth max: {}, min: {}".format(depth_image.max(), depth_image.min()))
    cv2.waitKey(50)


def callback(rgb_image, depth_image):
    global frameID
    print("Frame #{}".format(frameID))
    rgb_np_arr = np.fromstring(rgb_image.data, np.uint8)
    rgb_image_np = cv2.imdecode(rgb_np_arr, cv2.IMREAD_COLOR)
    
    depth_np_arr = np.fromstring(depth_image.data, np.uint8)
    depth_image_np = cv2.imdecode(depth_np_arr, cv2.IMREAD_GRAYSCALE )
    
    showImage(rgb_image_np, depth_image_np)
    frameID += 1
    
def main():
    
    rospy.init_node('image_receiver_all')

    rgb_image_sub = message_filters.Subscriber('/output/image_raw/compressed_rgb', CompressedImage)
    depth_image_sub = message_filters.Subscriber('/output/image_raw/compressed_depth', CompressedImage)

    ts = message_filters.TimeSynchronizer([rgb_image_sub, depth_image_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()

if __name__ == '__main__':
    main()