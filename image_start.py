#==================================================
#    Version : 2.0
#    Author  : muhz9786
#    Date    : 2019.05.22
#    Change  : 
#      Solved logical problem that results 
#      running extremely slow. 
#==================================================

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import time
import os
from PIL import Image

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
# Path of object_detection.
OD_PATH = './object_detection'
# Path of image dir.
IMAGES_DIR_NAME = 'test_image'

# Changes them base your data.
########################################
# Path of model and frozen graph.
# You can download some model in Detection Model Zoo,
# and place set model dir into the object_detection.
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
PATH_TO_MODEL =OD_PATH + '/' + MODEL_NAME    # Need not change.
PATH_TO_FROZEN_GRAPH = os.path.join(PATH_TO_MODEL, 'frozen_inference_graph.pb') 

# Path of label, generally need not to change.
PATH_TO_LABELS = OD_PATH + '/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90    # max number of label.           

# Path of image.
IMAGE_PATHS = [ os.path.join(IMAGES_DIR_NAME, '{}.jpg'.format(i)) for i in range(1, 7) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
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

# load image into numpy array.
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Get tensor form graph.
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Running in pre image.
        for image_path in IMAGE_PATHS:
            image = Image.open(image_path)    # Open image.

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
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

            # Display detection result.
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            plt.show()