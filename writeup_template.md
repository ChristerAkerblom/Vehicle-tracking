**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png
[image15]: ./output_images/noncar.png
[image2]: ./output_images/hog_car.png
[image25]: ./output_images/hog_noncar.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image500]: ./output_images/frame0.png
[image501]: ./output_images/heatmap0.png
[image510]: ./output_images/frame1.png
[image511]: ./output_images/heatmap1.png
[image520]: ./output_images/frame2.png
[image521]: ./output_images/heatmap2.png
[image530]: ./output_images/frame3.png
[image531]: ./output_images/heatmap3.png
[image540]: ./output_images/frame4.png
[image541]: ./output_images/heatmap4.png
[image550]: ./output_images/frame5.png
[image551]: ./output_images/heatmap5.png
[image60]: ./output_images/label0.png
[image61]: ./output_images/label1.png
[image62]: ./output_images/label2.png
[image63]: ./output_images/label3.png
[image64]: ./output_images/label4.png
[image65]: ./output_images/label5.png
[image7]: ./output_images/boxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image15]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color using Y channel space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2).

Here is an exampl of a car.

![alt text][image2]

Here is an example of a non car.

![alt text][image25]

#### 2. Explain how you settled on your final choice of HOG parameters.

I based my choice of parameters trying to balance both speed and accuracy of the CVM classifier using the data set.
I used `YCrCb` color space and channel Y for hog features. Color histogram and spatial features where also used.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the code cell with heading `Extract features for training classifier` of the IPython notebook.
I used both HOG and color features, spatial and histogram, for classification. 
A gridsearch method was used to find the best `C` parameter for the linear SVM classifier and a low `C=0.01` was used. The training test accuracy was `0.984`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the section `Processing pipeline (sub)` the sliding window search is implemented. HOG features are calculated once for the complete channel 0 following the project tips. The HOG features are combined with the spatial and histogram color features and then fed to the classifier.

For each frame the follwing regions were searched for vehicles ([y_start, y_stop, scale]):

`y_start_stop_scale = [[380,500,1.0], [380,550,1.5], [420,600,2.0], [450,680,3.0]]`

The image below shows the first attempt at using find_cars on one of the test images, using a single window size:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Windows 
Tracking

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4).
The vehicle tracking works pretty good. There are some detections of vehicles in the other side of the motorway, I have not considered them to be false detections ince they are actually detecting vehicles.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

In addition to that a vehicle tracking over frames was implemented in the list *trackingObjList*. For each detected vehicle the bounding box and a confidence measure was kept. For each new bounding box in a new frame the list was updated. If the new bounding box was close to an existing in the list the size, position and confidence measure of the box was updated. Every bounding box in *trackingObjList* with high enough confidence was displyed in the frame. Bounding boxes with to low confidence was deleted. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image500]
![alt text][image501]


![alt text][image510]
![alt text][image511]


![alt text][image520]
![alt text][image521]


![alt text][image530]
![alt text][image531]


![alt text][image540]
![alt text][image541]


![alt text][image550]
![alt text][image551]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image60]
![alt text][image61]
![alt text][image62]
![alt text][image63]
![alt text][image64]
![alt text][image65]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The filtering approach both in each frame by heatmaps and between frames by confidence measures was succesful. A better tuning of both classification parameters, search windows and filtering parameters will amke the file even more robust.

