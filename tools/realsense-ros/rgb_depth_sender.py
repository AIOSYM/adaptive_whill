#!/usr/bin/env python
"""
Read Images from realsense-camera and publish the RBG+Depth data
"""
import sys
import sys, time
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
import pyrealsense2 as rs
VERBOSE=False
try:
   sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
   pass

# __author__ =  'Khush Agrawal <agrawalkhush2000@gmail.com>'
import cv2
global stamp
stamp = 0

def pub_img():
    '''
    Callback function of subscribed topic.
    Here images get converted and features detected
    '''
    global stamp

    pipe = rs.pipeline()
    config = rs.config()
    width = 1280; height = 720;
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
    profile = pipe.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    pub_rgb = rospy.Publisher("/output/image_raw/compressed_rgb", CompressedImage, queue_size=1)
    pub_depth = rospy.Publisher("/output/image_raw/compressed_depth", CompressedImage, queue_size=1)

    print('RGB and Depth image are publishing.....')

    while(True):

        temp = pipe.wait_for_frames()
        aligned_frames = align.process(temp)
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            pass
        buffer = np.zeros((height,width,4))

        buffer[:,:,0:3] = (np.asanyarray(color_frame.get_data(),dtype=np.uint8))
        buffer[:,:,3] = np.asanyarray(aligned_depth_frame.get_data(),dtype=np.uint8)

        #### Create CompressedImage ####
        msg_rgb = CompressedImage()
        msg_rgb.header.stamp = rospy.Time.now()
        msg_rgb.format = "rgb"
        msg_rgb.data = np.array(cv2.imencode('.jpg', buffer[:,:,0:3])[1]).tostring()
        
        msg_depth = CompressedImage()
        msg_depth.header.stamp = msg_rgb.header.stamp
        msg_depth.format = "depth"
        msg_depth.data = np.array(cv2.imencode('.jpg', buffer[:,:,3])[1]).tostring()
        
        ## Publish RGB and Depth image as bytes of string 
        pub_rgb.publish(msg_rgb)
        pub_depth.publish(msg_depth)

        if rospy.is_shutdown():
            break

        stamp += 1
        if stamp >= 100000:
            stamp = 0

def main():
    '''Initializes and cleanup ros node'''
    rospy.init_node('image_feature')
    pub_img()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
