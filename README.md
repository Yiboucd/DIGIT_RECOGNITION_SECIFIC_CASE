# digit_recognition
 Digit recognition for a specific case
This project doesn't include auto number segment because the video is of pretty high quality, and if the camera is moved, you can adjust the predict function to make it suitable for your video. I have tested the result by myself, the accuracy is nearly 100 percent except for the case that the digit is changing just at the moment of screen shot, which is hard to avoid.
# Environment
Python 3.11; Numpy & Opencv-python are needed;

pip install numpy

pip install opencv-python

# Implementatiom
There are four entries in the GUI as shown in the figure below.
1. The first entry is the path of the video.
2. The second entry is the path of the sample, where you downdload, unzip and save. (This is the path of the folder!!!)
3. The third entry is where you want to save the screenshots, for the following detection. (This is the path of the folder!!!)
4. The fourth entry is the number of samplings that you want to obtain. (Positive integer only!!!)

![GUI](https://user-images.githubusercontent.com/130121873/230549289-ee368075-bd0c-4e95-87ab-b97e5f0a504c.png)

SOME IMPORTANT NOTES for implementation
1. The sample images can be downloaded, and make sure the sample images are PNG!!! (Because I use l2-norm, if it is not png, it will not work);
2. The path must be the path in your command prompt, and doesn't include "", like the figure as shown below.

![Path_Example](https://user-images.githubusercontent.com/130121873/230554751-e5bd5db3-682d-4b61-8c14-bf6bbc1154e5.png)

# Function Description & Adjustable parameters of the codes:
1. The screenshot function is used to sample the input video(given video_path) and save the screenshots to the save_path (This is a folder).
2. The l2_predict is used to predict the numbers on the images with the l2 norm between the part of input with the samples.
3. In this particular project, the segment of digits is by the values in codes as shwon below. If an auto-detection & auto-segment are needed, I can improve the codes. As so far, segment by precise values can greatly improve the accuracy, which is almost 100% regardless of the case that the screenshot just captures the digit changing.

![Adjustable Parameters](https://user-images.githubusercontent.com/130121873/230556272-9fc4852e-cb78-4aba-a6f6-cc437f5993bc.png)

4. In my experiment, the two given videos work perfectly, if the camera moves, you can just directly adjust the img_left & middle & right. In the case that the camera may move, I will adjust the codes, but the accuracy may be reduced.
