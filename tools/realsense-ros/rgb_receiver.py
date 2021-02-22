#!/usr/bin/env python
import cv2 
import rospy
import message_filters
import numpy as np

from sensor_msgs.msg import CompressedImage

frameID = 0

def rgb_callback(rgb_image):
    
    global frameID
    print("Frame #{}".format(frameID))
    rgb_np_arr = np.fromstring(rgb_image.data, np.uint8)
    rgb_image_np = cv2.imdecode(rgb_np_arr, cv2.IMREAD_COLOR)
    cv2.imshow("RGB image", cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR))
    frameID += 1
    cv2.waitKey(1)
    
def main():
    
    rospy.init_node('image_receiver_rgb')

    rospy.Subscriber('/output/image_raw/compressed_rgb', CompressedImage, rgb_callback)
    rospy.spin()

if __name__ == '__main__':
    main()