import cv2 as cv

def abs_rel(predicted, ground):
    """
    Parameters:
    predicted { array(array(int)) } 
        A 2D array containing a predicted depth map. This is generally the result from applying a depth estimation model.
    ground { array(array(int)) } 
        A 2D array containing the ground truth depth map. Should be representing the same image used in prediction.

    Output:
    absrel { float }
        A float representing the absolute relative error of the data.
        Returns None if the array dimensions of `predicted` does not match `ground`.
    """

    if len(predicted) != len(ground) or len(predicted[0]) != len(ground[0]):
        print("WARNING: Skipped evaluation of AbsRel due to inconsistent sizes between predicted and ground truth values.\n"
              f"{len(predicted)}, {len(predicted[0])} in predicted, {len(ground)}, {len(ground[0])} in ground truth.")
        return None
    
    absrel = 0
    for i in range(len(ground)):
        for j in range(len(ground[0])):
            absrel += abs(predicted[i][j] - ground[i][j]) / ground[i][j]
    absrel /= (len(ground) * len(ground[0]))

    # absrel = sum([abs(predicted[i][j] - ground[i][j]) / ground[i][j] for j in range(len(ground[0])) for i in range(len(ground))]) / (len(ground) * len(ground[0]))
    return absrel

def rmse(predicted, ground):
    """
    Parameters:
    predicted { array(array(int)) } 
        A 2D array containing a predicted depth map. This is generally the result from applying a depth estimation model.
    ground { array(array(int)) } 
        A 2D array containing the ground truth depth map. Should be representing the same image used in prediction.

    Output:
    rmse { float }
        A float representing the root mean square error of the data.
        Returns None if the array dimensions of `predicted` does not match `ground`.
    """

    if len(predicted) != len(ground) or len(predicted[0]) != len(ground[0]):
        print("WARNING: Skipped evaluation of RMSE due to inconsistent dimensions between predicted and ground truth values.\n"
              f"{len(predicted)}, {len(predicted[0])} in predicted, {len(ground)}, {len(ground[0])} in ground truth.")
        return None
    
    se_total = 0
    for i in range(len(ground)):
        for j in range(len(ground[i])):
            error = predicted[i][j] - ground[i][j]
            se_total += error ** 2
    mse = se_total / (len(ground) * len(ground[0]))
    rmse = mse ** 0.5

    # rmse = (sum([(predicted[i][j] - ground[i][j]) ** 2 for j in range(len(ground[0])) for i in range(len(ground))]) / (len(ground) * len(ground[0]))) ** 0.5
    return rmse

def dbe_accuracy(predicted, ground):
    if len(predicted) != len(ground):
        print("WARNING: Skipped evaluation of dbe_accuracy due to inconsistent sizes between predicted and ground truth values.\n"
              f"{len(predicted)} in predicted, {len(ground)} in ground truth.")
        return None

    # Requires edge maps
    predicted_edges = cv.Canny(predicted, 100, 200)
    ground_edges = cv.Canny(ground, 100, 200)

    # Euclidean Distance Transform
    ground_dist = cv.distanceTransform(ground_edges, cv.DIST_L2, 5)

    # Calculate DBE
    for i in range(predicted):
        for j in range(predicted[i]):
            pass


def dbe_completeness(predicted, ground):
    if len(predicted) != len(ground):
        print("WARNING: Skipped evaluation of dbe_completeness due to inconsistent sizes between predicted and ground truth values.\n"
              f"{len(predicted)} in predicted, {len(ground)} in ground truth.")
        return None
    
    return dbe_accuracy(ground, predicted)
