# computerVision-greenScreenRemoval
This project was part of my opencv.org computer vision course.
The goal was to implement a system which can replace the background of a green screen video with an arbitrary image.

My solution is as follows:

In a first step various background patches on the first frame of the video can be selected. Using these samples,
 HSV ranging and morphological operations a trimap is created dividing the frame in to a certain background area,
 a certain foreground area and a region in between where it is not known for sure whether it belongs to the foreground
 or the background (uncertain area).

 The next step is to build the alpha mask for the uncertain area. For each pixel the mean of the certain foreground and the certain
 background neighborhood is evaluated. Given the eucledean distance between these two means the probability that the pixel
 belongs to the foreground or the background is evaluated using using a logistic cumulative distribution function with
 a mean located at half the distance between the mean foreground and the mean background. The sigma depends on the users
 input for softness.

 Now the edges are soft. But the problem of green background bleeding into the edge area still remains. To replace these
 colors the certain foreground neighborhood of the pixel is evaluated. It is replaced by the intensity of the pixel in
 the certain foreground neighborhood with the shortest euclidean distance.

 Having replaced the green pixels and calculated the alpha mask the new background image is multiplied with the inverse alpha mask
 and added to the frame multiplied by the alpha mask

 Of course today there are much more sophisticated ways for green screen removal. This is just something I tried out.
 It works quite well. To improve it the neighborhood could be circular instead of squares. Moreover the color correction near the edges could be improved.
 It always tends to take light parts from the foreground because the edge pixels where the green light shines through are light as well.
 This could be improved by taking into account the background color in the neighborhood.