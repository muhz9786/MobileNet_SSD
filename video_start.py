#==================================================
#    Version : 3.0
#    Author  : muhz9786
#    Date    : 2019.05.29
#    Change  : 
#      Rewrote the code that converts image into numpy array,   
#      which results running slowly. 
#==================================================

import numpy as np
import tensorflow as tf
import cv2
import os
import time
from PIL import Image

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
# Path of object_detection.
OD_PATH = './object_detection'
# Path of video dir.
VIDEO_DIR_NAME = 'test_video'

# Changes them base your data.
########################################
# Path of model and frozen graph.
# You can download some model in Detection Model Zoo,
# and place set model folder into the object_detection.
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
PATH_TO_MODEL =OD_PATH + '/' + MODEL_NAME    # Need not change.
PATH_TO_FROZEN_GRAPH = os.path.join(PATH_TO_MODEL, 'frozen_inference_graph.pb') 

# Path of label, generally need not to change.
PATH_TO_LABELS = OD_PATH + '/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90    # max number of label.           

# Select video path.
VIDEO_PATH = os.path.join(VIDEO_DIR_NAME, 'road1.mp4')   
# VIDEO_PATH = 0    # use camera.              

# Set optimization of OpenCV.
# cv2.setUseOptimized(True)     # I'm not sure it is helpful.
########################################

# Load a frozen model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

mv = cv2.VideoCapture(VIDEO_PATH)    # open video

gpu_options = tf.GPUOptions(allow_growth=True)    # allocate GPU memory based on requirement of computing.

with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Get tensor form graph.
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Running in pre frames.
        while True:
            return_value, frame = mv.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                raise ValueError("No image!") 

            prev_time = time.time()     # get start time.

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = np.array(image, dtype=np.uint8)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=6)

            # Counts running time for pre frame and display it.
            # You can comment out them if you do not need.
            curr_time = time.time()     # get end time.
            exec_time = curr_time - prev_time
            info = "time: %.2f ms" %(1000*exec_time)
            cv2.putText(image_np, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=2)

            # Display detection result.
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            
            # You can press "q" to exit when running.
            if cv2.waitKey(1) & 0xFF == ord('q'):    
                break