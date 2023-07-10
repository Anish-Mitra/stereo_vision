ENPM673 Project3
Anish Mitra (amitra12@umd.edu)

This folder contains all the necessary python scripts required to perform stereo vision on the dataset given to us. The directory structure of the project file is as follows

amitra12_proj3
			|
			|- curule --
			|			|- calib.txt
			|			|- im0.png
			|			|- im1.png 
			|- octagon--
			|			|- calib.txt
			|			|- im0.png
			|			|- im1.png
			|- pendulum--
			|			|- calib.txt
			|			|- im0.png
			|			|- im1.png
			|
			|- calibration.py
			|- rectification.py
			|- correspondence.py
			|- compute_depth.py
			|- pipeline.py
			|- stereo_vision.py

Curule, Octagon, Pendulum folders are the dataset required to perform stereo vision. 

calibration.py,rectification.py,correspondence.py,compute_depth.py and pipeline.py are the necessary helper files required to perform stereo vision.


stereo_vision.py is the main executable python file that can be run at the command terminal to perform stereo vision on one of the data set.

Run the command "python3 stereo_vision.py" in the command terminal and select the required data set.

NOTE: It is NECESSARY that the dataset folder be present in the same file as all the other python scripts.