#!/usr/bin/env python3
import rospy
import torch
import numpy as np
import PIL
import os
import argparse

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from cv_bridge import CvBridge

from tools.wrapper_yolo_base import YoloWrapper, DEVICE, example_use_yolo
from tools.msgpacking import init_matrix_array_ros_msg, pack_multiarray_ros_msg

from tools.wrapper_yolo_base import plot_one_box, color_list
from tools.sort import Sort



YOLO_MODEL = YoloWrapper('yolo_param/yolov5s.pt', DEVICE)
YOLO_MODEL.model.eval()
YOLO_MODEL.model.to(DEVICE)
FREQ_NODE = 10


### ROS Subscriber Callback ###
IMAGE_RECEIVED = None
def fnc_img_callback(msg):
    global IMAGE_RECEIVED
    IMAGE_RECEIVED = msg



def run_yolo_node(args):

    mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    colors = color_list()

    # rosnode node initialization
    rospy.init_node('perception_node')   # rosnode node initialization
    print("Perception_node is initialized at", os.getcwd())

    # subscriber init.
    sub_image = rospy.Subscriber('/carla_node/cam_down/image_raw', Image, fnc_img_callback)   # subscriber init.
    

    # publishers init.
    pub_yolo_prediction = rospy.Publisher('/yolo_node/yolo_predictions', Float32MultiArray, queue_size=10)   # publisher1 initialization.
    pub_yolo_boundingbox_video = rospy.Publisher('/yolo_node/yolo_pred_frame', Image, queue_size=10)   # publisher2 initialization.
    pub_sort_prediction = rospy.Publisher('/yolo_node/sort_mot_predictions', Float32MultiArray, queue_size=10)   # publisher1 initialization.
    pub_sort_boundingbox_video = rospy.Publisher('/yolo_node/sort_mot_frame', Image, queue_size=10)    # publisher3 initialization.
    
    rate=rospy.Rate(FREQ_NODE)   # Running rate at 20 Hz

    # a bridge from cv2 image to ROS image
    mybridge = CvBridge()

    # msg init. the msg is to send out numpy array.
    msg_mat = init_matrix_array_ros_msg()
    t_step = 0

    ##############################
    ### Instructions in a loop ###
    ##############################
    while not rospy.is_shutdown():

        t_step += 1

        if IMAGE_RECEIVED is not None:
            np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)
            np_im = np.array(np_im)

            with torch.no_grad():
                x_image = torch.FloatTensor(np_im).to(DEVICE).permute(2, 0, 1).unsqueeze(0)/255
                cv2_images_uint8, prediction_np = YOLO_MODEL.draw_image_w_predictions(x_image.detach())

            # Filter Car or Truck
            try:
                tgt_boxes = prediction_np[np.where( prediction_np[:,5]==2 )]  # label 2 is car and label 5 is truck.
            except:
                tgt_boxes = []

            # Use SOT to track objects.
            if len(tgt_boxes) > 0:
                trackers = mot_tracker.update(tgt_boxes[:,:5])  # Bayesian Update
            else:
                trackers = mot_tracker.update(np.empty((0, 5))) # Prediction

            # Draw the tracked detections.
            if len(trackers) > 0:
                for d in trackers:
                    d = d.astype(np.int32)
                    plot_one_box((d[0],d[1],d[2],d[3]), np_im, color=colors[1], label=str(d[4]%100))
            
            # Publish the image with the tracked detections.
            image_message = mybridge.cv2_to_imgmsg(np_im, encoding="passthrough")
            image_message.encoding = "rgb8"
            pub_sort_boundingbox_video.publish(image_message)

            try:
                pub_sort_prediction.publish(pack_multiarray_ros_msg(msg_mat, trackers))      
            except:
                pub_sort_prediction.publish(pack_multiarray_ros_msg(msg_mat, np.array([[1]])))      
            
            
            ### Publish the prediction results in results.xyxy[0]) ###
            #                   x1           y1           x2           y2   confidence        class
            # tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
            #         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
            #         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])
            try:
                pub_yolo_prediction.publish(pack_multiarray_ros_msg(msg_mat, prediction_np))      
            except:
                pub_yolo_prediction.publish(pack_multiarray_ros_msg(msg_mat, np.array([[1]]))) 


            ### Publish the bounding box image ###
            image_message = mybridge.cv2_to_imgmsg(cv2_images_uint8, encoding="passthrough")
            image_message.encoding = "rgb8"
            pub_yolo_boundingbox_video.publish(image_message)

        rate.sleep()


def main():
    # example_use_yolo()

    parser = argparse.ArgumentParser(description='ROS YOLO NODE')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=10)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)    

    args, unknown = parser.parse_known_args()

    try:
        run_yolo_node(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')




if __name__ == '__main__':

    try:
        main()
    except rospy.ROSInterruptException:
        pass

    
    
    

    
    