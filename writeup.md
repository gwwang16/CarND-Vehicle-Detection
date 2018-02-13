## Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_example.png
[image3]: ./output_images/boxes.png
[image4]: ./output_images/heatmap.png
[image5]: ./output_images/video_result.png

[video1]: ../project_video_output.mp4

---
There are two main files: `Vehicle_detection.ipynb` and `Vehicle_detection_pipeline.ipynb`

`Vehicle_detection.ipynb` is used for illustrating detailed processes,

`Vehicle_detection_pipeline.ipynb` is used for generating video. 

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is in `Load Dataset` and `HOG Visulization` sections of `Vehicle_detection.ipynb`.

- Read all image paths using 

```
images = glob.glob("../data/*/*/*.png")
cars = []
notcars = []
for image in images:
    if 'non-vehicles' in image:
        notcars.append(image)
    else:
        cars.append(image)
```

These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/). Here is an example of  `vehicle` and `non-vehicle` 

![alt text][image1]

- Explore different color spaces and hog parameters with trial and error

  The following parameters are finally selected
```
feature_params = {'orient': 11,
                  'pix_per_cell': 16,
                  'cell_per_block': 2,
                  'hog_channel': 'ALL',
                  'spatial_size': (32,32),
                  'hist_bins': 32,
                  'spatial_feat': False,
                  'hist_feat': True,
                  'hog_feat': True,
                  'color_space': 'YUV'}
```

Here is an example using GRAY color space its HOG image:


![alt text][image2]

#### 2. Describe how you trained a classifier using your selected HOG features.

The code for this step is in `Extract Features` and `Train Classifier` sections of `Vehicle_detection.ipnb`.

The gradient feature and HOG features are extracted using `extract_features()` ,and then they are standardized  using  [`sklearn.preprocessing.StandardScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) package to remove the mean and scale to unit variance.

Then, I trained a linear SVM using [`sklearn.svm.LinearSVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn-svm-linearsvc) package, the accuracy of 97.47% is achieved on test dataset.

### Sliding Window Search

#### 1. Describe how you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is in `Sliding Windows` section of `Vehicle_detection.ipynb`.

`find_cars()`  extracts hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells.  Then the SVM classifier predicts whether it contains a car nor not in each window. To generate multiple-scaled search windows, different scales can be combined with each other. 

After trying some combinations, scales [1.3, 1.8] and `cells_per_step = 1, a window overlap of 75%, are settled considering with false positives and calculation speed. 

```
def multi_scale(img, scales, svc, X_scaler, feature_params):
    draw_img = np.copy(img)
    box_lists = []
    for scale in scales:
        if scale < 1.5:
            ystart = 400
            ystop = 580
        else:
            ystart = 400
            ystop = 660
        box_list = find_cars(img, scale, svc, X_scaler, feature_params, ystart, ystop)
        box_lists.append(box_list)
        for box in box_list:
            cv2.rectangle(draw_img,box[0],box[1],(0,0,255),5)
    return draw_img, box_lists
```

Here are outputs of all 6 test images using multiple scales.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

There are lots of false positives and multiple detections from above image. To optimize its performance, heatmap is adopted. 

For each box in the boxlist, add 1 for all pixels inside each box to generate heat map.

```
def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap
```

Zero out all pixels below the threshold, 4 is used in this project.

```
def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap
```

Then use `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap, and assume each blob corresponded to a vehicle.  Most of false positives are filtered using the following code

```
def draw_heat_box(img, box_lists, heat_threshold):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    for box_list in box_lists:
        heat = add_heat(heat, box_list)
    heatmap_thresh = apply_threshold(heat, heat_threshold)
    heatmap = np.clip(heatmap_thresh, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img, heatmap
```

 Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Despite of the heatmap threshold method, I also used `collections.deque` to store past 10 frames, and compute the mean values of all these heatmaps .
```
    def draw_heat_box(self, img, box_lists, heat_threshold):
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        for box_list in box_lists:
            heat = self.add_heat(heat, box_list) 
        self.heatmaps.append(heat)
        heatmap_avr = sum(self.heatmaps)/10.
        heatmap_thresh = self.apply_threshold(heatmap_avr, heat_threshold)
        heatmap = np.clip(heatmap_thresh, 0, 255)
        labels = label(heatmap)
        draw_img = self.draw_labeled_bboxes(np.copy(img), labels)
        return draw_img
```
A judgement condition is also added to draw box on the image only when its area, width and height are larger than thresholds.
```
    def draw_labeled_bboxes(self, img, labels):
        """Draw labeled boxes"""
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            area_threshold = 300
            w_threshold = 50
            h_threshold = 50
            box_w = np.max(nonzerox)-np.min(nonzerox)
            box_h = np.max(nonzeroy)-np.min(nonzeroy)
            box_area = box_w * box_h
            if (box_area > area_threshold) and (box_w > w_threshold) and (box_h > h_threshold):
                # Draw the box on the image if its area larger than threshold
                cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 8)
        return img
```



Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image5]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are still many problems in this project. 

- The calculation speed is too slow (about 2sec/frame) for real implementation. It;s said neural network works well for this project, I will implement it in the future.
- There are still a few false positives, different color spaces combinations and threshold method would be investigated further.
-  ROI method might be useful to speed up calculation time



This is the last day of this course, it's an interesting project, but I have no time to explore it further, have to submit it now. Hope to receive more comments to improve it in the future.