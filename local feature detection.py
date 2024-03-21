import os
import cv2
import numpy as np
from scipy.ndimage import sobel, gaussian_filter, maximum_filter
from scipy.spatial.distance import cdist

def HarrisPointsDetector(image, threshold=0.7):
    # Apply Sobel operator
    Ix = sobel(image, axis=1, mode='reflect')
    Iy = sobel(image, axis=0, mode='reflect')

    # Gaussian filter
    sigma = 0.5
    Ix2 = gaussian_filter(Ix ** 2, sigma)
    Iy2 = gaussian_filter(Iy ** 2, sigma)
    Ixy = gaussian_filter(Ix * Iy, sigma)

    # Compute Harris response
    det_M = Ix2 * Iy2 - Ixy ** 2
    trace_M = Ix2 + Iy2
    R = det_M - 0.05 * trace_M ** 2

    # Find local maxima as key points
    local_maxima = maximum_filter(R, size=7)
    keypoints = np.argwhere((R == local_maxima) & (R > threshold))
    
    # Compute orientations
    orientations = np.degrees(np.arctan2(Iy, Ix))
    orientations[orientations < 0] += 180

    return keypoints, orientations


def featureDescriptor(image, keypoints):
    orb = cv2.ORB_create()
    #keypoints_cv2 = [cv2.KeyPoint(x=point[1], y=point[0], _size=20) for point in keypoints]
    #keypoints_cv2 = orb.detect(image,None)
    keypoints_cv2, descriptors = orb.detectAndCompute(image, None)

    return descriptors



def SSDFeatureMatcher(descriptor1, descriptor2):
    distances = cdist(descriptor1, descriptor2, 'sqeuclidean')
    matches = []
    for i, dist in enumerate(distances):
        idx1 = i
        idx2 = np.argmin(dist)
        distance = dist[idx2]
        matches.append(cv2.DMatch(idx1, idx2, 0, distance))
    return matches

def RatioFeatureMatcher(descriptor1, descriptor2, ratio=0.8):
    distances = cdist(descriptor1, descriptor2, 'sqeuclidean')
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

# Load reference image
reference_image = cv2.imread('bernieSanders.jpg', cv2.IMREAD_GRAYSCALE)

# Detect interest points using custom Harris detector
reference_keypoints, reference_orientations = HarrisPointsDetector(reference_image)
ref_keypoints_cv2 = [cv2.KeyPoint(x=float(point[1]), y=float(point[0]), size=10) for point in reference_keypoints]

# Compute descriptors using ORB compute function from OpenCV
reference_descriptors = featureDescriptor(reference_image, ref_keypoints_cv2)

# Specify the directory containing the other images
directory = 'COMP37212'

# Load other images
other_images = []
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
        other_images.append(img)


# Loop over other images
for i, image in enumerate(other_images):
    # Detect interest points using custom Harris detector
    keypoints, _ = HarrisPointsDetector(image)
    
    # Convert keypoints to cv2.KeyPoint objects
    keypoints_cv2 = [cv2.KeyPoint(x=float(point[1]), y=float(point[0]), size=10) for point in keypoints]
    
    # Compute descriptors using ORB compute function from OpenCV
    descriptors = featureDescriptor(image, keypoints_cv2)
    
    # Match features
    #matches = SSDFeatureMatcher(reference_descriptors, descriptors)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(reference_descriptors,descriptors)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    # Create an empty image with the same dimensions as one of the input images
    output_image = np.zeros((max(reference_image.shape[0], image.shape[0]), reference_image.shape[1] + image.shape[1], 3), dtype=np.uint8)

    # Draw matches on the output image
    matched_image = cv2.drawMatches(reference_image, ref_keypoints_cv2, image, keypoints_cv2, matches[:10], None)

    
    # Visualize matches
    # draw_params = dict(matchColor=(0, 0, 255), 
    #                singlePointColor = None,
    #                flags = 0
    #               )
    # matched_image = cv2.drawMatches(reference_image, ref_keypoints_cv2, image, keypoints_cv2, matches, None, 2)
    
    # Save or display matched image
    cv2.imwrite(f'matched_image_{i+1}.jpg', matched_image)
    # cv2.imshow(f'matched_image_{i+1}', matched_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
