import cv2 
import yaml
import sys 

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

def get_tracker_configuration(config_file):
	## Open config file 
	with open(config_file) as f:
		try:
			config = yaml.safe_load(f)
		except yaml.YAMLError as e:
			print(e)
			sys.exit() 
	tracker_object = get_tracker(config['TRACKER_KEY'])
	
	return tracker_object

def get_tracker(tracker_key):
    return OPENCV_OBJECT_TRACKERS[tracker_key]