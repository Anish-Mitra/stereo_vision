import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to perform correspondance and calculate disparity map using SSD
def ssd_correspondence(rectified_img1, rectified_img2):
    
    rect1_grey = cv2.cvtColor(rectified_img1, cv2.COLOR_BGR2GRAY)
    rect2_grey = cv2.cvtColor(rectified_img2, cv2.COLOR_BGR2GRAY)
    # Window size on second image
    disparities = 50
    # Pixel size of block in first image
    block_size = 15
    height, width = rect1_grey.shape
    disparity_map = np.zeros(shape=(height, width))
    print("Computing disparity map using SSD")
    # Compute disparity map using SSD by comparing matches along epipolar lines
    for i in range(block_size, rect1_grey.shape[0] - block_size - 1):
        for j in range(block_size + disparities, rect1_grey.shape[1] - block_size - 1):
            SSD = np.empty([disparities, 1])
            l = rect1_grey[(i - block_size):(i + block_size), (j - block_size):(j + block_size)]
            height, width = l.shape
            for d in range(0, disparities):
                r = rect2_grey[(i - block_size):(i + block_size),(j - d - block_size):(j - d + block_size)]
                SSD[d] = np.sum((l[:, :]-r[:, :])**2)
            disparity_map[i, j] = np.argmin(SSD)
    # Rescale SSD to 0-255
    disparity_map_scaled = ((disparity_map/disparity_map.max())*255).astype(np.uint8)
    
    return disparity_map_scaled


   

   



