# MobileNet SSD

---
Implementation of MobileNet SSD by Tensorflow Object Detection API.  
You can do detection both on **image** and **video**.

---
## Status

- ***Date:*** 2019.05.29
- ***Version:*** 3.0
- ***Model:*** ssd_mobilenet_v2_coco_2018_03_29
- ***Training Status:***
    ```
    (No training)
    ```
---
## Data Location

- **Image:**  
`./test_image`  
- **Video:**  
`./test_video`

---
## Start Deteceion

- **Image Dectection:**  
`python ./image_start.py`
- **Video Detection:**  
`python ./video_start.py`  
If use camera, set `VIDEO_PATH = 0` in code (`start_video.py`).

---
*Thanks to Google and Contributors of Tensorflow.*