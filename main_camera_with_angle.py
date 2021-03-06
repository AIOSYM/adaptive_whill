#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

import cv2 
import numpy as np

from sensor.realsense.setup import get_camera_configuration
from detector.YOLOv3.detector import get_detector_configuration, yolo_output
from tracker.opencv.tracker import get_tracker_configuration
from controller.pid_controller import get_controls, stop_controls
from utils.util import points_perspective_transform, get_average_distance, get_coordinates, filter_bbox_based_on_distance

## ----------------------------------
# Initialize new parameters
IS_TRACKING = False
x,y,w,h = 0,0,0,0
REFRESH_INTERVAL = 50
min_range = 0  # (m)
max_range = 2  # (m)
FRAME_ID = 0
i_error_l = 0
i_error_a = 0
d_error_l = 0
d_error_a = 0
new_size = (720, 480)
## Homography Metrix (for image correction with around 30 degrees angle)
H = np.array([[ 2.66835209e+00/1.3, -3.79148708e-02, -0.19699952e+03],
 [ 4.78788512e-01,  2.14900022e+00/1.3, -4.31157942e+02],
 [ 1.22350798e-03, -3.96466173e-05,  1.00000000e+00]])
## ----------------------------------

## Setup sensor
pipeline, align = get_camera_configuration()

## Setup detector
model, data, confidence, nms_thesh, inp_dim, CUDA = get_detector_configuration('configs/yolov3.yaml')

## Setup tracker
tracker_object = get_tracker_configuration('configs/opencv-tracker.yaml')

## Create new ROS node to publish /cmd_vel
rospy.init_node('adaptive')

## Create publisher to publish twist msg
pub = rospy.Publisher('/whill/controller/cmd_vel', Twist, queue_size=10)

## Create publisher to publish rgb image
pub_rgb = rospy.Publisher("/output/image_raw/compressed_rgb", CompressedImage, queue_size=1)
msg_rgb = CompressedImage()

## Runing Loop
while True:
    try:
        ## Update frame id 
        FRAME_ID += 1

        ## Align depth and color frame
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        
        ## Get the color frame 
        color_frame = frames.get_color_frame()
        rgb_img = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        
        ## WarpPerspective
        rgb_warp = cv2.warpPerspective(rgb_img, H, (rgb_img.shape[1], rgb_img.shape[0]))

        ## Get the depth frame
        depth_frame = frames.get_depth_frame()
        depth_img = np.asanyarray(depth_frame.get_data(), dtype=np.uint8)

        if not color_frame or not depth_frame:
            continue

        if FRAME_ID%REFRESH_INTERVAL == 0 or not IS_TRACKING:
            
            ## YOLO: Detect person to follow 
            print('NEW DETECTION')
            yolo_frame = rgb_img.copy()
            img, bbox = yolo_output(yolo_frame, model, data, ['person'], confidence, nms_thesh, CUDA, inp_dim)

            # Filter the person to follow based on distance
            person_in_range_bbox = filter_bbox_based_on_distance(bbox, depth_frame, min_range, max_range)
            
            ## Create new tracker
            print('NEW TRACKER')
            initBB, trueBB = get_coordinates(person_in_range_bbox, x, y, x+w, y+h)
            tracker = tracker_object()
            tracker.init(rgb_img, initBB)
        
        (IS_TRACKING, box) = tracker.update(rgb_img)
        
        if IS_TRACKING:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(rgb_img, (x, y), (x + w, y + h),(0, 255, 0), 2)
            
            # Transform the points
            x1, y1, x2, y2 = x, y, x+w, y+h
            x1, y1, x2, y2 = points_perspective_transform(x1, y1, x2, y2, H)
            cv2.rectangle(rgb_warp,(x1, y1), (x2, y2),(0, 255, 0), 2)

            # Use transform points
            xt, yt, wt, ht = int(x1), int(y1), int(x2-x1), int(y2-y1)
        
            if x < 0 or y < 0:
                twist = stop_controls()
            else:
                calc_x, calc_z = (xt+wt/2), get_average_distance(depth_frame, (x,y, x+w, y+h))
                twist, error = get_controls(calc_x, calc_z, 1/3, 0, 0.2,-1/500, 0, 0, i_error_l, i_error_a, d_error_l, d_error_a)
                print(f'linear x: {twist.linear.x}, angular z: {twist.angular.z}')
                
                i_error_l, i_error_a, d_error_l, d_error_a = error
        else:
            print("STOP THE WHEEL CHAIR------------")
            twist = stop_controls()

        ## Display 
        rgb_resize = cv2.resize(rgb_img, new_size, interpolation=cv2.INTER_AREA)
        #cv2.imshow('RGB_FRAME', cv2.cvtColor(rgb_resize, cv2.COLOR_RGB2BGR))
        rgb_warp = cv2.resize(rgb_warp, new_size, interpolation=cv2.INTER_AREA)
        cv2.imshow('Wrap_FRAME', cv2.cvtColor(rgb_warp, cv2.COLOR_RGB2BGR))
        yolo_frame = cv2.resize(yolo_frame, new_size, interpolation=cv2.INTER_AREA)
        #cv2.imshow('YOLO_FRAME', cv2.cvtColor(yolo_frame, cv2.COLOR_RGB2BGR))

        #### Create CompressedImage ####   
        msg_rgb.header.stamp = rospy.Time.now()
        msg_rgb.format = "rgb"
        msg_rgb.data = np.array(cv2.imencode('.jpg', rgb_warp)[1]).tostring()

        pub_rgb.publish(msg_rgb)

        ## Exit program sequence
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or rospy.is_shutdown():
            print('shutdown')
            break
        if key == ord("s"):
            print('save image ')
            cv2.imwrite('saved_{}.jpg'.format(FRAME_ID),cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print("Error occured!!")
        print(e)
        twist = stop_controls()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or rospy.is_shutdown():
            print('shutdown')
            break

    finally:
        print("PUBLISHING...")
        pub.publish(twist)
        
pipeline.stop()
cv2.destroyAllWindows()
exit(0)

