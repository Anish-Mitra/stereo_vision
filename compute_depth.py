import numpy as np
import cv2

# Function to compute depth map from disparity map
def disparity_to_depth(baseline, focal_length, disparity_map_scaled):
    
    depth=np.zeros(shape=disparity_map_scaled.shape).astype(float)
    depth[disparity_map_scaled > 0] = (focal_length * baseline) / (disparity_map_scaled[disparity_map_scaled > 0])
    depth_map = ((depth/depth.max())*255).astype(np.uint8)
    
    return depth_map
