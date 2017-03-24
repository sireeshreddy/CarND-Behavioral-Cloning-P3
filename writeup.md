# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Nvidia-architecture.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model I used was the nvidia model as suggested in the lectures. The only difference was the input image used was 160x320x3 instead of the 66x200x3 in the nvidia model.

The first layer of the model has a lambda layer which normalizes the input. Followed by that, the image is cropped to ignore the hood of the car as well as the eliminate some of the sky, trees etc (using a Cropping layer to eliminate areas that may not be of significance in training the model).The rest of the model follows the nvidia model.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 
The data was split 80/20 (training/validation) to avoid overfitting. For testing, the model was tested on the simulator in autonomous mode and was observed to stay on track through the entire course.ehicle could stay on the track.

#### 3. Model parameter tuning

The model was tuned using a mean squared loss method and the adam optimizer.

#### 4. Appropriate training data

After following much of the discussion on the slack channels I decided to go with the test data that came with the project as a lot of people had success with it. Considering a lot of the students had trouble generating their own usable data it seemed prudent to go with the test data and avoid introducing another point of failure.

During development and training of the model I tried using some user generated data which actually performed flawlessly on the "Jungle track". It however was not as accurate on the primary track and I decided to go with the test set. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The initial approach to the problem was to implement the recently tested and proven Nvidia end to end deep learning model for sel-driving cars. This took out a lot of guesswork from the equation allowing me to spend more time in other areas of the project. The model seemed to work quite well beginning with the test runs so I decided to just add on top of it the lambda layer and cropping layer (both ideas that were explained in the course lectures)

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. (80 % - Training and 20% - Validation). 

To avoid overfitting, 2 dropout layers of 50% were added.
As explained in this paper, http://papers.nips.cc/paper/4878-understanding-dropout.pdf that seemed adequate to ensure we did not overfit the model

The final step was to run the simulator to see how well the car was driving around track one. The car seemed to drive almost flawlessly through the track except for the are right after the bridge which does not have a clearly marked right hand side barrier. My initial thought was this was because of the training data bias. After trying out a few different sample data sets the problem persisted. There was marginal difference with increasing the data size or modifying parameters within the model and even increasing the number of epochs.
In the end, applying a gamma transformation inspired by this article
http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/ made navigating the "problem" portion of the track possible.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

As explained aboce, the final model architecture was a slightly modified Nvidia architechture with a larger input frame of 160x320x3 followed by a lambda layer and a cropping layer. 

#### 3. Creation of the Training Set & Training Process

There were 2 possible approches to the training data. Either use the simulator to drive around the track and obtain training data or use the test set provided. With other students having mixed success with their own training data, I decided to go with the Udacity test set.
The training data was pre processed and converted to YUV space as that worked the best with the Nvidia model (as described in the above referenced paper) To rule out any (left-turn)bias on the test set, the data was augmented from all 3 camera images by adding mirror images to the test set. This brought the total number of training images to 7 times the initial set (~45k samples) and with plenty of data to train the model on.

Although even with a small dataset using only the center camera image in the YUV space was fairly successful, there was one location in the course, where the car consistently went off track. Ultimately, the fix for that was to apply gamma transformation to the images. 
What was interesting to note, is applying that transformation made the model drive very poorly on the 2nd "Jungle Track" while improving driving behavior on the main track. Since that was not crucial to completion of this project I did not spend a lot of time analyzing this but it is definitely of interest to understand this better.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. After testing out epochs in the range of 5-20, a total number of epochs = 15 seemed optimal. I used an adam optimizer so that manually training the learning rate wasn't necessary.
