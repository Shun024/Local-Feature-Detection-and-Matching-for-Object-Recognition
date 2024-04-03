# Local Feature Detection

**Coursework Objective:**

The objective of this coursework is to implement local feature detection and matching algorithms for object recognition. Specifically, the task involves detecting local features in an image using the Harris corner detection method, describing these features using the ORB (Oriented FAST and Rotated BRIEF) descriptor, and then matching these features between different images.

**Learning Outcomes:**

1. **Understanding of Local Feature Detection Algorithms**: gain an understanding of local feature detection algorithms, specifically the Harris corner detection method, and its implementation.
2. **Feature Descriptor Implementation**: learn to implement feature descriptors such as ORB to describe local features detected in images.
3. **Feature Matching Techniques**: learn different techniques for matching features between images, including sum of squared differences (SSD) and ratio test.
4. **Programming Skills in Image Processing**: Through the implementation of these algorithms using NumPy, SciPy, and OpenCV functions, enhance their programming skills in image processing and computer vision.

**Summary of Coursework Tasks:**

1. **Feature Detection (Harris Corner Detection)**: Implement the Harris corner detection method to identify interest points in the image. Compute the Harris matrix for each point using Sobel operators for derivative calculation and Gaussian mask for image padding. Then compute the corner strength function and select local maxima as interest points.

2. **Feature Description (ORB Descriptor)**: Use the ORB descriptor to create descriptors for the features centered at each interest point detected in the previous step. This descriptor will serve as the representation for comparing features in different images.

3. **Feature Matching**: Implement feature matching functions such as sum of squared differences (SSD) and ratio test to find the best matching features between images. Compare features from the reference image (bernieSanders.jpg) with those from other provided images.

4. **Visualizing and Reporting Results**: Visualize the detected interest points and matches between images. Experiment with different parameter choices and analyze their effects on the results. Benchmark the algorithm's performance using a set of benchmark images provided.

**Implementation and Report Submission Details:**

1. Implement the HarrisPointsDetector function for interest point detection, the featureDescriptor function using ORB for feature description, and the matchFeatures function for feature matching (both SSD and ratio test).

2. Submit a report including benchmarking results, parameter choices, quantitative results, and observations. Compare the performance of ORB features using Harris interest points detector versus ORB features using FAST interest point detector based on experimental results and observations.

One of the many results:

![bernieBenefitBeautySalon_rf_harris_matches](https://github.com/Shun702/Local-Feature-Detection-and-Matching-for-Object-Recognition/assets/138392252/564a51dd-3053-4376-bb4b-51d12d4fec35)

Checkout the results folder for all the result images...
