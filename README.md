# MobileNet SSD

---
Implementation of MobileNet SSD by Tensorflow Object Detection API.  
You can do detection both on **image** and **video**.

---
## Status

- ***Date:*** 2019.05.22
- ***Version:*** 2.0
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
`python ./start_image.py`
- **Video Detection:**  
`python ./start_video.py`  
If use camera, set `VIDEO_PATH = 0` in code (`start_video.py`).

---
*Thanks to Google and Contributors of Tensorflow.*