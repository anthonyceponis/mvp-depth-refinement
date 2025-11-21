import numpy as np
import cv2 as cv

def abs_rel(predicted: np.array, ground: np.array):
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
    print(ground)
    absmap = np.abs(predicted - ground)
    absmap = absmap / ground
    absrel = np.sum(absmap) / ground.size

    return absrel

def rmse(predicted, ground):
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
    rmse = (np.sum(error ** 2) / ground.size) ** 0.5

    return rmse

def dbe_accuracy(predicted, ground):
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
    predicted_edges = cv.Canny(predicted, 100, 200)
    ground_edges = cv.Canny(ground, 100, 200)

    # Rescale predicted_edges so it is between 0 and 1
    predicted_edges = predicted_edges / 255

    # Euclidean Distance Transform
    ground_dist = cv.distanceTransform(ground_edges, cv.DIST_L2, 5)

    print(f"Predicted Edges:\n{predicted_edges}\n\nGround EDT:\n{ground_dist}")

    # Calculate depth boundary accuracy error
    dbe_acc = np.sum(ground_dist * predicted_edges) / np.sum(predicted_edges)
    return dbe_acc

def dbe_completeness(predicted, ground):
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
    dbe_comp = dbe_accuracy(ground, predicted)

    return dbe_comp

# Test evaluations
# Create test depth maps
cv.imwrite("./test_images/test_prediction.jpg", 
    np.array([
    [100, 100, 100, 100, 100, 1],
    [100, 100, 100, 100, 1, 100],
    [100, 100, 100, 1, 100, 100],
    [100, 100, 1, 100, 100, 100],
    [100, 1, 100, 100, 100, 100],
    [1, 100, 100, 100, 100, 100]
    ], dtype=np.uint8)
)
cv.imwrite("./test_images/test_ground_truth.jpg", 
    np.array([
    [100, 100, 100, 100, 50, 10],
    [100, 100, 100, 50, 10, 50],
    [100, 100, 50, 10, 50, 100],
    [100, 50, 10, 50, 100, 100],
    [50, 10, 50, 100, 100, 100],
    [10, 50, 100, 100, 100, 100]
    ], dtype=np.uint8)
)
cv.imwrite("./test_images/test_erroneous.jpg", 
    np.array([
    [1, 1, 1, 1, 5, 9],
    [1, 5, 9, 9, 5, 1],
    [9, 5, 1, 1, 1, 1]
    ], dtype=np.uint8)
)

# Reimport the images
p = cv.imread("./test_images/test_prediction.jpg")
gt = cv.imread("./test_images/test_ground_truth.jpg")
err = cv.imread("./test_images/test_erroneous.jpg")

# Evaluate on predefined arrays
print(f"""
AbsRel: {abs_rel(p, gt)}
RMSE: {rmse(p, gt)}
DBE_acc: {dbe_accuracy(p, gt)}
DBE_comp: {dbe_completeness(p, gt)}
""")

# Try errorneous examples
# Should log skipping evaluation due to error
abs_rel(p, err)
rmse(err, gt)
dbe_accuracy(p, err)
