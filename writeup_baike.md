# CarND Vehicle Detection Project

## Goals

The goals/steps of this project are the following:

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
- Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
- Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!
All of the code for the project is contained in the [CarND Vehicle Detection notebook](Vehicle_detection_BK.ipynb).

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

My first step is to load all of the vehicle and non-vehicle image paths from the provided dataset. A random sample of images from both car and non-car dataset are shown below, respectively:

![Roandom Car images](./misc/Car_Visualization.JPG)

![Roandom Car images](./misc/NonCar_Visualization.JPG)

The total nummber of the training images is:

- Vehicle train images count: 8792
- Non-vehicle train image count: 8968

The feature extraction code (spatial, color and HOG) is contained In cell `[4]` of [CarND Vehicle Detection notebook](Vehicle_detection_BK.ipynb). By using the method of ` get_hog_features `, a comparsion of a car image and its associated histogram of oriented gradients, as well as the same for a non-car image are shown as below:

![HOG Vehicle and non-vehicle images](./misc/comparsion_Hog_features.JPG)

In the cell of `[6]` of [CarND Vehicle Detection notebook](Vehicle_detection_BK.ipynb), a function of extract_features  is to accept a list of image paths, spatial information, Histogram and HOG parameters ( as well as one of a variety of destination color spaces, to which the input image is converted) and produces a flattened array of spatial feature, histogram feature, or/and HOG features as defined for each image in the list.

#### 2. Explain how you settled on your final choice of HOG parameters.

The parameters were found by manually changing them and experimenting. My final choice of HOG parameters based upon the performance of the SVM classified produced using them. Not only the accuracy with which the classiefier made prediction on the test data, but also the speed at which the classifier is able to make predictions have been considered. The final parameters are the following:

|Parameter|Value|
|:--------|----:|
|Color Space|YCrCb|
|HOG Orient|11|
|HOG Pixels per cell|8|
|HOG Cell per block|2|
|HOG Channels|All|
|Spatial bin size| (16,16)|
|Histogram bins|32|
|Histogram range|(0,256)|
|Classifier|LinearSVC|
|Scaler|StandardScaler|

With this parameters, it took 130.83 seconds to get features and 17.63 seconds to train SVC.

#### 3.  Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the section titled "Train a Classifier" ( cell of `[8]` of [CarND Vehicle Detection notebook](Vehicle_detection_BK.ipynb) ). I trained a linear SVM with the default classifier parameters and using spatial intensity, chanell intensity histogram features , and HOG features and was able to achieve a test accuracy of 99.41%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

My first approach to implement sliding windows was to calculate all the windows and then apply the feature extraction to each one of them to find the one containing a car. It is implemented on `In [8]`. Cells `In [9]` and `In [10]` contains the code for loading the test images, applying the classifier to the images and drawing boxes. The scales and overlap parameter where found by experimenting on them until a successful result was found. The following image shows the results of this experimentation on the test images:

![Sliding windows first implementation](images/sliding_windows.png)

To combine the boxes found there and eliminate some false positives, a heat map as implemented with a threshold and the function `label()` from `scipy.ndimage.measurements` was used to find where the cars we. The code for this implementation could be found on `In [13]`, and the next image shows the results on the test images:

![Sliding windows with heatmap and threshold](images/withheatmap.png)

#### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

The performance of the method calculating HOG on each particular window was slow. To improve the processing performance, a HOG sub-sampling was implemented as suggested on Udacity's lectures. The implementation of this method could be found on `In [14]`. The following image shows the results applied to the test images (the same heatmap and threshold procedure was applied as well on `In [15]`):

![Sliding windows with HOG sub-sampling](images/hog_subsampling.png)

### Video implementation

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The video output could be found [project_video.mp4](video_output/project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some filter for false positives and some method for combining overlapping bounding boxes.

Some effort was done already to minimize false positives using a heatmap and threshold in the pipeline, but it was not enough. The overlapping bounding boxes were resolved by using the function `label()` from `scipy.ndimage.measurements` to find the cars. To filter false positives, the image heatmap map was averaged over three consecutive frames. The implementation could be found on `In [25]`

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

- Use a decision tree to analyze the redundancy on the feature vector and try to decrease its length eliminating redundancy.

- The performance of the pipeline could be improved by trying to decrease the amount of space to search for windows.

- More than one scale could be used to find the windows and apply them on the heatmap.

- The windows size could change for different X and Y values to minimize the number of windows to process.
