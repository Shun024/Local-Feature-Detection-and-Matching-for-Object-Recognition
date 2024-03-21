import os
import sys
import cv2
import numpy as np
import matplotlib as plt
from scipy import ndimage, spatial

INPUT_DIRECTORY = "./COMP37212/"
OUTPUT_DIRECTORY = "./results/"

"""
image: mat
- image must be grayscale
sigma: int
- for gausian blurring (default value of 0.5)
alpha: int
- used for computing the corner strength funtion, R
threshold: int
- calcualte the non-maxima supression (optional)
nms: int
- window size on both axes (default value of 7)
"""
def HarrisPointsDetector(
        image, sigma=0.5, alpha=0.05, threshold_value= 0.05, nms=7, graph=False
):
    # Blur gray image
    blur_image = ndimage.gaussian_filter(image, sigma=10, mode="reflect", radius=2)

    # Normalize the image to np.float64
    image_64 = blur_image.astype(np.float64)
    image_64 = blur_image / 255.0

    # Compute image gradients in X and Y directions; axis = 0 for y derivative, axis = 1 for x derivative
    x_gradient_64 = ndimage.sobel(image_64, axis=1, mode="reflect")
    y_gradient_64 = ndimage.sobel(image_64, axis=0, mode="reflect")

    # Compute the combinations between the gradients
    orientations = np.rad2deg(np,arctan2(y_gradient_64, x_gradient_64))

    # Compute the cominations between the gradients for M Matrix: (I_x)^2, (I_y)^2, (I_x)(I_y)
    x2_gradient_64 = np.square(x_gradient_64)
    y2_gradient_64 = np.square(y_gradient_64)
    xy_gradient_64 = np.multiply(x_gradient_64, y_gradient_64)

    # Blur the combinations between the gradients
    blur_x2_gradient_64 = ndimage.gaussian_filter(x2_gradient_64, sigma=sigma, mode="reflect", radius=2)
    blur_y2_gradient_64 = ndimage.gaussian_filter(y2_gradient_64, sigma=sigma, mode="reflect", radius=2)
    blur_xy_gradient_64 = ndimage.gaussian_filter(xy_gradient_64, sigma=sigma, mode="reflect", radius=2)

    # Compute the 2x2 M Matrix Determinant and Trace 
    determinant = np.subtract( np.multiply(blur_x2_gradient_64, blur_y2_gradient_64), np.square(blur_xy_gradient_64))
    trace = np.add(x2_gradient_64, y2_gradient_64)

    # Compute the corner strength function, 𝑅
    r = determinant - (alpha * np.square(trace))

    maxes = ndimage.maximum_filter(r, size=nms, cval=1e100)
    r_maxes = maxes == r
    rows, cols = image_64.shape

    # Loop through all feature points in r_maxes, where each poiont is filled with content required for descriptor computation
    # requires x, y, and angle
    features = []
    for y in range(rows):
        for x in range(cols):
            if not r_maxes[y, x]:
                continue

            f = cv2.KeyPoint(x, y, 7, orientations[y][x], r[y][x])
            features.append(f)
    
    """ Graph production when graph flag is true
    graph showing relation between number of keypoints and threshold value
    """
    if graph:
        thresholds = [0.01 * i for i in range(1, 20)]
        no_kps = [0] * len(thresholds)

        for i in range(len(thresholds)):
            threshold = np,max(r) * thresholds[i]
            if features is not None:
                keypoints = [ keypoint for keypoint in features if keypoint.response >= threshold
                ]
                no_kps[i] = len(keypoints)

        plt.plot(thresholds, no_kps)
        plt.xlabel("Thresholds")
        plt.ylabel("Number of Keypoints")
        plt.title("Thresholds vs Number of Keypoints")
        plt.savefig("./threshold_vs_keypoints.jpg")
    
    # Find the threshold
    # If the response value of the keypoint is greater than threshold, return the keypoint, else don't
    threshold = np.max(r) * threshold
    if features is not None:
        keypoints = [ keypoint for keypoint in features if keypoint.response >= threshold
        ]

    return keypoints

def featureDescriptor(image, keypoints):
    _, descriptor = orb.compute(image, keypoints)

    return descriptor

def SSDFeatureMatcher(ref_desc, sample_desc):
    if ref_desc.shape[0] == 0 or sample_desc[0] == 0:
        return []
    distances = spatial.distance.cdist(ref_desc, sample_desc, 'sqeuclidean')
    matches = []
    for i, dist in enumerate(distances):
        idx1 = i
        idx2 = np.argmin(dist)
        distance = dist[idx2]
        matches.append(cv2.DMatch(idx1, idx2, 0, distance))
    return matches

"""
Uses a ratior of SSD distance of the two best matches and 
matches a feature in the first image with the cloeset feature in the second image.

NOTE: There is possibility of having the same multiple features from the first image with the second image"""
def RatioFeatureMatcher(ref_desc, sample_desc, ratio=0.75):
    if ref_desc[0] == 0 or sample_desc[0] == 0:
        return []
    distances = cdist(ref_desc, sample_desc, 'sqeuclidean')
    matches = []
    for i, dist in enumerate(distances):
        idx1 = i
        sorted_indices = np.argsort(dist)
        best_match_dist = dist[sorted_indices[0]]
        second_best_match_dist = dist[sorted_indices[1]]
        if best_match_dist < ratio * second_best_match_dist:
            idx2 = sorted_indices[0]
            matches.append(cv2.DMatch(idx1, idx2, 0, best_match_dist))
    return matches

if __name__ == "__main__":

    # Open beernieSanders.jpg as reference image 
    reference_image = cv2.imread("bernieSanders.jpg", cv2.IMREAD_COLOR)

    # Check if the reference image exists
    if reference_image is None:
        print("Error: Failed to open the required image (bernieSanders.jpg")
        sys.exit(1)

    # Convert the colored referenced image to gray
    reference_gray_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    ## ORB Built in Points Detector ##
    # Initialize ORB object
    orb = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE)
    # Find interest points using OpenCV corner detector
    orb_kps= orb.detect(reference_image, None)
    # Compute the descriptors with ORB interest points
    orb_kps, orb_desc = orb.compute(reference_image, orb_kps)
    # Draw keypoints
    gray_kps = cv2.drawKeypoints(reference_gray_image,orb_kps, None, color=(0, 0, 255), flags=0)
    # Output image for comparison discussion in report
    cv2.imwrite(OUTPUT_DIRECTORY + "orb_built_in_features.jpg", gray_kps.astype(np.uint8))

    ## Own Harris Points Detector Implementation ##
    harris_ref_kps = HarrisPointsDetector(reference_gray_image, graph=True)
    harris_ref_image = cv2.drawKeypoints( reference_gray_image, harris_ref_kps, None, color=(0, 0, 255), flags=0)

    # Find descriptor using the keypoints from own Harris Points Detector
    harris_ref_desc = featureDescriptor(reference_gray_image, harris_ref_kps)

    # Output image for comparison discussion in report
    cv2.imwrite(OUTPUT_DIRECTORY + "harris_implementation_features.jpg", harris_ref_kps.astype(np.uint8))


    for file in os.listdir(INPUT_DIRECTORY):
        if file == ".DS_Store":
            continue

        # Open image
        image_sample = cv2.imread(INPUT_DIRECTORY + file, cv2.IMREAD_COLOR)

        # Convert image to gray
        image_sample_gray = cv2.cvtColor(image_sample, cv2.COLOR_BGR2GRAY)

        if "benieMoreblurred" in file:
            harris_sample_kps = HarrisPointsDetector(image_sample_gray)
        else: 
            harris_sample_kps = HarrisPointsDetector(image_sample_gray)
        
        # Draw keypoints
        harris_sample_image = cv2.drawKeypoints( image_sample_gray, harris_sample_kps)
        harris_sample_desc = featureDescriptor(image_sample_gray, harris_sample_kps)

        ## Find matching feature using SSDFeatureMatcher ##
        ssd_matches = sorted( SSDFeatureMatcher(harris_ref_desc, harris_sample_desc), key=lambda x: x.distance, )
        ssd_img_match = cv2.drawMatches(
            harris_ref_image, harris_ref_kps, harris_sample_image, harris_sample_kps, ssd_matches, None, flags=2,
        )

        name = file.split(".")[0] + "_ssd_harris_matches.jpg"
        cv2.imwrite(OUTPUT_DIRECTORY + name, ssd_img_match)


        ## Find matching feature using RatioFeatureMatcher ##
        rf_matches = sorted( RatioFeatureMatcher(harris_ref_desc, harris_sample_desc), key=lambda x: x.distance, )
        rf_img_match = cv2.drawMatches(
            harris_ref_image, harris_ref_kps, harris_sample_image, harris_sample_kps, rf_matches, None, flags=2,
        )

        name = file.split(".")[0] + "_rf_harris_matches.jpg"
        cv2.imwrite(OUTPUT_DIRECTORY + name, rf_img_match)

            

