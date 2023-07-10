This folder contains all the necessary Python scripts required to perform stereo vision on the dataset given to us. The directory structure of the project file is as follows

```bash
.
├── curule/
│   ├── calib.txt
│   ├── im0.png
│   └── im1.png 
├── octagon/
│   ├── calib.txt
│   ├── im0.png
│   └── im1.png
├── pendulum/
│   ├── calib.txt
│   ├── im0.png
│   └── im1.png
├── Project_description.pdf
├── calibration.py
├── rectification.py
├── correspondence.py
├── compute_depth.py
├── pipeline.py
└── stereo_vision.py
```

<br>Curule, Octagon, and Pendulum folders are the dataset required to perform stereo vision.
<br>
<br>`calibration.py`, `rectification.py`, `correspondence.py`, `compute_depth.py`, and `pipeline.py` are the necessary helper files required to perform stereo vision.
<br>
<br>`stereo_vision.py` is the main executable python file that can be run at the command terminal to perform stereo vision on one of the datasets.
<br>
<br>Run the command `python3 stereo_vision.py` in the command terminal and select the required dataset.
<br>
**NOTE:** It is **NECESSARY** that the dataset folder be present in the same file as all the other Python scripts.
