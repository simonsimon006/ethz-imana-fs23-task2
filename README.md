# ethz-imana-fs23-task2
Task two, image segementation, for the course Image Analysis and Computer Vision.

## Task Description
The task is to segement a given image into a foreground and a background depending on a users choice, i.e. create a segmentation mask. For the given image we receive two sets of pixels:
- The foreground set where we are guaranteed that all pixels in it belong to the users choice of foreground.
- The background set with an analoguous gurantee for the chosen background.

## Solution Outline
First we use take the foreground set and view all its pixels as 3-vectors [R, G, B]. Additionally, we use a self-implemented fft-convolution with multiple averaging filters to find the neighbourhoods average intensities for all three colours. These values are also appended to the pixel vector. After this we use k-means clustering to find k_fg clusters in the foreground set and k_bg in the background set. We can find these hyperparameters via GridSearch on a validation set. Now we apply a similar processing to our full image (i.e. 3-vec + Neighbour Intensities) and check for each pixel if the cluster closest to it is in either the foreground cluster set or the background cluster set. If it is the former, we set the masks value to True at the coordinates of the pixel, otherwise we set it to False.

## Implemetation Details
- I did not implement the check for cluster-closeness per-pixel, instead I computed all the distances at once and selected the argmin, i.e. the index of the smallest cluster in bulk.
- For the convolution I used numpys rfft2 and irfft2. The (constant-) averaging kernel was always quadratic but had varying sizes. It was zero-padded to fit the image size.
