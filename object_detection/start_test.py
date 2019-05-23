import os
import cv2
import time
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.append('..')
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Select video path 
VIDEO_PATH = "./test_video/road.mp4"

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Reshape image
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def detect_objects(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image})

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
        return image_np

    
vid = cv2.VideoCapture(VIDEO_PATH)
while True: 
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
    else:
        raise ValueError("No image!")
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Counts runing time
    # prev_time = time.time()
    # Actual detection.

    prev_time = time.time()

    result = detect_objects(image_np_expanded, detection_graph)
    # Visualization of the results of a detection.

    curr_time = time.time()
    exec_time = curr_time - prev_time
    info = "time: %.2f ms" %(1000*exec_time)

    cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255, 0, 0), thickness=2)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)

    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# cv2.destroyAllWindows()
