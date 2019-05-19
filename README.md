# MobileNet SSD
---
Implementation of MobileNet SSD in Tensorflow Object Detection API.

---
## Status
- ***Date:*** 2019.5.19
- ***Version:*** 1.0
- ***Model:*** ssd_mobilenet_v2_coco_2018_03_29
- ***Training Status:***
    ```
    (No training)
    ```
---
## Data Location
- **Image:**
`./object_detection/test_image`  
Please use `.jpg` image.
- **Video:**
`./object_detection/test_video`

---
## Start Deteceion
- **Image Dectection:**  
`python ./object_detection/start_image.py`
- **Video Detection:**  
`python ./object_detection/start_video.py`  
If use camera, set `VIDEO_PATH = 0` in code (`start_video.py`).

---
*Thanks to Google and Contributors of Tensorflow.*