import numpy as np
import cv2 as cv
from ppd_sharpdepth.ppd.utils.depth2pcd import depth2pcd
from ppd_sharpdepth.sharpdepth.util.alignment import align_depth_least_square

def auto_canny_depth_otsu(depth_map, apertureSize=3, L2gradient=False, dilate_kernel=0, low_frac=0.5):
    """
    Automatic Canny edge detection using Otsu thresholding on gradient magnitudes.

    Parameters:
        depth_map : 2D numpy array (depth values, any range)
        apertureSize : Sobel kernel size (3,5,7)
        L2gradient : whether to use L2 norm for gradient magnitude
        dilate_kernel : optional, size of square kernel to dilate edges after detection
        low_frac : fraction of high threshold to use as low threshold (default 0.5)

    Returns:
        edges : binary edge map (uint8, 0 or 255)
        threshold1, threshold2 : thresholds used
    """
    # 1. Normalize depth map to 0-255 uint8
    depth_norm = np.clip(depth_map, np.min(depth_map), np.max(depth_map))
    depth_uint8 = ((depth_norm - np.min(depth_norm)) / (np.max(depth_norm) - np.min(depth_norm)) * 255).astype(np.uint8)

    # 2. Compute gradient magnitude using Sobel
    grad_x = cv.Sobel(depth_uint8, cv.CV_64F, 1, 0, ksize=apertureSize)
    grad_y = cv.Sobel(depth_uint8, cv.CV_64F, 0, 1, ksize=apertureSize)
    
    if L2gradient:
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    else:
        grad_mag = np.abs(grad_x) + np.abs(grad_y)

    # 3. Otsu threshold on gradient magnitude
    grad_uint8 = np.clip((grad_mag / grad_mag.max() * 255), 0, 255).astype(np.uint8)
    threshold2, _ = cv.threshold(grad_uint8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Convert to native Python floats
    threshold2 = float(threshold2)
    threshold1 = float(low_frac * threshold2)

    # 4. Run Canny
    edges = cv.Canny(depth_uint8, threshold1, threshold2,
                     apertureSize=apertureSize, L2gradient=L2gradient)

    # 5. Optional dilation
    if dilate_kernel and dilate_kernel > 0:
        kernel = np.ones((dilate_kernel, dilate_kernel), dtype=np.uint8)
        edges = cv.dilate(edges, kernel)

    return edges

def abs_rel(predicted: np.ndarray, ground: np.ndarray, valid_mask: np.ndarray):
    """
    Parameters:
    predicted { np.array(ndim=2) } 
        A 2D numpy array containing a predicted depth map. This is generally the result from applying a depth estimation model.
    ground { np.array(ndim=2) } 
        A 2D numpy array containing the ground truth depth map. Should be representing the same image used in prediction.

    Output:
    absrel { float }
        A float representing the absolute relative error of the data.
        Returns None if the array dimensions of `predicted` does not match `ground`.
    """

    # Ensure predicted and ground have matching dimensions.
    if predicted.shape != ground.shape:
        print("WARNING: Skipped evaluation of AbsRel due to inconsistent sizes between predicted and ground truth values.\n"
              f"{predicted.shape} in predicted, {ground.shape} in ground truth.")
        return None
    
    # Prevent any divide by zero errors
    predicted += 1
    ground += 1
    
    # Calculate the absolute relative error
    absmap = np.abs(predicted - ground)
    absmap[~valid_mask] = 0.0
    absmap = absmap / ground
    absrel = np.sum(absmap) / np.sum(valid_mask) 

    return absrel

def rmse(predicted: np.ndarray, ground: np.ndarray, valid_mask: np.ndarray):
    """
    Parameters:
    predicted { np.array(ndim=2) } 
        A 2D numpy array containing a predicted depth map. This is generally the result from applying a depth estimation model.
    ground { np.array(ndim=2) } 
        A 2D numpy array containing the ground truth depth map. Should be representing the same image used in prediction.

    Output:
    rmse { float }
        A float representing the root mean square error of the data.
        Returns None if the array dimensions of `predicted` does not match `ground`.
    """

    # Ensure predicted and ground have matching dimensions.
    if predicted.shape != ground.shape:
        print("WARNING: Skipped evaluation of RMSE due to inconsistent dimensions between predicted and ground truth values.\n"
              f"{predicted.shape} in predicted, {ground.shape} in ground truth.")
        return None
    
    # Calculate the root mean square error
    error = predicted - ground
    error[~valid_mask] = 0.0
    rmse = (np.sum(error ** 2) / np.sum(valid_mask)) ** 0.5

    return rmse

def dbe_accuracy(predicted: np.ndarray, ground: np.ndarray, valid_mask: np.ndarray):
    """
    Parameters:
    predicted { np.array(ndim=2) } 
        A 2D numpy array containing a predicted depth map. This is generally the result from applying a depth estimation model.
    ground { np.array(ndim=2) } 
        A 2D numpy array containing the ground truth depth map. Should be representing the same image used in prediction.
    
    Output:
    dbe_acc { float }
        A float representing the depth boundary accuracy error of the data.
        Returns None if the array dimensions of `predicted` does not match `ground`.
    """

    # Ensure predicted and ground have matching dimensions.
    if predicted.shape != ground.shape:
        print("WARNING: Skipped evaluation of dbe_accuracy due to inconsistent sizes between predicted and ground truth values.\n"
              f"{predicted.shape} in predicted, {ground.shape} in ground truth.")
        return None

    # Requires edge maps 
    predicted_edges = auto_canny_depth_otsu(predicted, dilate_kernel=5)
    ground_edges = auto_canny_depth_otsu(ground, dilate_kernel=5)
 
    # Euclidean Distance Transform
    ground_dist = cv.distanceTransform(ground_edges, cv.DIST_L2, 5)

    # Apply valid mask.
    #predicted_edges *= valid_mask
    #ground_dist *= valid_mask

    #print(f"Predicted Edges:\n{predicted_edges}\n\nGround EDT:\n{ground_dist}")

    # Calculate depth boundary accuracy error
    dbe_acc = np.sum(ground_dist * predicted_edges) / np.sum(predicted_edges)
    return dbe_acc, predicted_edges, ground_edges

def dbe_completeness(predicted: np.ndarray, ground: np.ndarray, valid_mask: np.ndarray):
    """
    Parameters:
    predicted { np.array(ndim=2) } 
        A 2D numpy array containing a predicted depth map. This is generally the result from applying a depth estimation model.
    ground { np.array(ndim=2) } 
        A 2D numpy array containing the ground truth depth map. Should be representing the same image used in prediction.
    
    Output:
    dbe_comp { float }
        A float representing the depth boundary completeness error of the data.
        Returns None if the array dimensions of `predicted` does not match `ground`.
    """

    # Ensure predicted and ground have matching dimensions.
    if predicted.shape != ground.shape:
        print("WARNING: Skipped evaluation of dbe_completeness due to inconsistent sizes between predicted and ground truth values.\n"
              f"{predicted.shape} in predicted, {ground.shape} in ground truth.")
        return None
    
    # Calculate the depth bounary completeness error by calling the accuracy error function with reversed arguments 

    # Normalize to 0–255, because canny maps work on 8bit grayscale imgs.
    predicted = predicted - predicted.min()
    if predicted.max() > 0:
        predicted = predicted / predicted.max()
    predicted = (predicted * 255).astype(np.uint8)

    ground = ground - ground.min()
    if ground.max() > 0:
        ground = ground / ground.max()
    ground = (ground * 255).astype(np.uint8)


    dbe_comp, predicted_edges, ground_edges = dbe_accuracy(predicted, ground, valid_mask)

    return dbe_comp, predicted_edges, ground_edges

def ppd_metric(pred_depth: np.ndarray, gt_depth: np.ndarray, intrinsic: np.ndarray):

    least_squares_pred_depth, _, _ = align_depth_least_square(
        gt_arr=gt_depth,
        pred_arr=pred_depth,
        valid_mask_arr=np.ones_like(pred_depth,dtype=np.bool),
        return_scale_shift=True,
        max_resolution=None,
    )

    dilate_kernel=5 # width of region around edges.
    ground_edges = auto_canny_depth_otsu(gt_depth, dilate_kernel=dilate_kernel)

    ground_edges = ground_edges.reshape(-1).astype(bool) # flatten and cast.

    pred_point_cloud = depth2pcd(least_squares_pred_depth, intrinsic, ret_pcd=True, input_mask=ground_edges)
    gt_point_cloud = depth2pcd(gt_depth, intrinsic, ret_pcd=True, input_mask=ground_edges)

    d1 = np.asarray(gt_point_cloud.compute_point_cloud_distance(pred_point_cloud))
    d2 = np.asarray(pred_point_cloud.compute_point_cloud_distance(gt_point_cloud))

    scene_scale = (gt_depth.max() - gt_depth.min())

    chamfer_dist = (np.mean(d1) + np.mean(d2)) / scene_scale

    return chamfer_dist





