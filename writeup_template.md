#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

I have included the following files :
1. model.py -> contains conde to load data , preprocess images and train the model using CNN.
2. drive.py -> A script provided which was used to record the driving in autonomous mode .
3. model.h5 -> Trained model
4. model.mp4    -> Video of car completing one track completely . 



####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5

Submission includes model.h5 and other script/files as mentioned above.

```

####3. Submission code is usable and readable

Yes , I have attached model.py and model.h5 file which can be run in any other machine . Also I have added comments in model.py file
so that code is readable and each part of the code is explained why it is put there with its functionality .

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

To train I used Nvidia's model where I used convolution layers with 5*5 filter sizes
and depths of 100 , 50 and 10 . I used relu function to introduce non linearltiy . 
Apart from that I have normalized the data , cropped the images ( so as to avoid GIGO - Garbage
In Gargbage Out)  


####2. Attempts to reduce overfitting in the model

Earlier I tried with normal convolutional layer , but later on I switched to Nvidia
model which was giving me better results.  In order to reduce overfitting I suffled the data,
flipped the images , added left and right images and also I added recovery data so that
there is no overfitting . 


####3. Model parameter tuning

I used adam optimizer and as such for model parameter tuning only thing which
I did was increase the number of epochs. 

####4. Appropriate training data

In order to train the model I started with sampling data which was provided . 
However I felt this was not enough so I added recoverry data especially for 
curves where I had to drive the car quite a few times and collect that data
and add recovery images to given IMAGES and add recovery metadata to 
driving_log.csv

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use CNN . 

My first step was to use a convolution neural network model as follows: 

model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape = (160,320 ,3)))
model.add(Convolution2D(6,5,5,activation ='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5,activation ='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1))

However , using model trained using above CNN was making the car go out of track. 
I read in blogs and slack that lot of people have tried using NVIDIA architecture
so then i switched to NVIDA architecture as follows : 

-- Implement the Nvidia model using sub sampling , introducing non linearity (relu)
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,5,5, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



In order to gauge how well the model was working, I split my image and steering 
angle data into a training and validation set with 80% training set and 20% as testing set

To combat the overfitting, I used flipping , croping the images (avoid GIGO - Garbage In
Garbage out) , using left and right images . I also used recovery data especially around the 
curves .

The final step was to run the simulator to see how well the car was driving
around track one. At the end of the process, the vehicle is able to drive autonomously 
around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 94-103) consisted of a Nvidia's convolution neural network with the following layers and layer sizes :


model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,5,5, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



####3. Creation of the Training Set & Training Process

*Apology I am not able to embed images , hence I have attached the images in folder
submission and also posted links of aws-s3 where I have uploaded for your convinence*

Initially I took the dataset from the sample repository which was provided by Udacity . There were approx 8K images each for Center
Left and right images . Those images were as follows : 


####Image set 1 
Center Image 1 : 
https://s3.ap-south-1.amazonaws.com/udacityrrelan/main/center_2016_12_01_13_30_48_287.jpg

Left Image 1 : 
https://s3.ap-south-1.amazonaws.com/udacityrrelan/main/left_2016_12_01_13_30_48_287.jpg

Right Image 1: 
https://s3.ap-south-1.amazonaws.com/udacityrrelan/main/right_2016_12_01_13_30_48_287.jpg



#### Image Set 2
Center Image 2 :
https://s3.ap-south-1.amazonaws.com/udacityrrelan/main/center_2016_12_01_13_31_12_937.jpg

Left Image 2 :
https://s3.ap-south-1.amazonaws.com/udacityrrelan/main/left_2016_12_01_13_31_12_937.jpg

Right Image 2: 
https://s3.ap-south-1.amazonaws.com/udacityrrelan/main/right_2016_12_01_13_31_12_937.jpg

#### HISTOGRAM OF STEERING ANGLE OF SAMPLE DATA PROVIDED 
Using the above data set my data's steering angle distribution was as shown in the
following histogram : 
https://s3.ap-south-1.amazonaws.com/udacityrrelan/Histogram/Sample_Training_Data.png


#### WHY I WENT FOR RECOVERY MODEL 
When I trained my model , my car crashed on the second turn which was quite sharp
and required sharp steering angles. So in order to get through the second sharp
turn I captured recovery data across this 2nd curve where I drove past this curve
around 4 times and also recovered the data by moving from edges of the curve 
to the center of the curve (I ensured that I capture sharp steering angles as
the car has very short time to make a sharp turn hence the need for high steering 
angles) 

Following are sample recovery images which I recorded  :

Recovery Center Image 1 :
https://s3.ap-south-1.amazonaws.com/udacityrrelan/recv/recv_center_2017_04_16_20_53_20_205.jpg

Recvoery Left Image 1 : 
https://s3.ap-south-1.amazonaws.com/udacityrrelan/recv/recv_left_2017_04_16_20_53_20_205.jpg

Recvoery Right Image 1 : 
https://s3.ap-south-1.amazonaws.com/udacityrrelan/recv/recv_right_2017_04_16_20_53_20_205.jpg

####RECOVERY DATASET HISTOGRAM
Using the above data set my data's steering angle distribution was as shown in the
following histogram for RECOVERY DATASET: 
https://s3.ap-south-1.amazonaws.com/udacityrrelan/Histogram/Recovery_Data_Histogram.png


##DATA AUGMENTATION :
I also flipped the images for data generalization . Here are the flipped 
images of main data set and recovery data set : 

###Flipped Data - For main data set

Center 1 :
https://s3.ap-south-1.amazonaws.com/udacityrrelan/main_flip/aug_center_1.jpeg

Left 1 :
https://s3.ap-south-1.amazonaws.com/udacityrrelan/main_flip/aug_left_1.jpeg

Right 1 :
https://s3.ap-south-1.amazonaws.com/udacityrrelan/main_flip/aug_right_1.jpeg


Center 2 :
https://s3.ap-south-1.amazonaws.com/udacityrrelan/main_flip/aug_center_2.jpeg

Left 2 :
https://s3.ap-south-1.amazonaws.com/udacityrrelan/main_flip/aug_left_2.jpeg

Right 2 :
https://s3.ap-south-1.amazonaws.com/udacityrrelan/main_flip/aug_right_2.jpeg

###Flipped Data - For RECOVERY DATA SET 

Recovery Center Image :
https://s3.ap-south-1.amazonaws.com/udacityrrelan/recv_flip/recv_aug_center_1.jpeg

Recovery Left Image :
https://s3.ap-south-1.amazonaws.com/udacityrrelan/recv_flip/recv_aug_left_1.jpeg

Recovery Right Image :
https://s3.ap-south-1.amazonaws.com/udacityrrelan/recv_flip/recv_aug_right_1.jpeg



And the histogram of recovery images is as follows which shows a lot of data 
with negative steering angles this is because 2nd turn was quite sharp and 
to go pass through it I had to capture recovery images around it which
finally led to my model performing successful lane drive across the track :) 



To summarize what I did till now , to capture good driving behavior, I used sampling data which was provided (having 
approx. 24 K images , 8 k images each from left center and right images) . 
Apart from that I collected recovery data especially among the curves (around 1k 
data for each left right and center images) . Thus I trained my model on 
27k images . Over these 27k images I also flipped them to generalize the model 
and adding correction of 0.25 for steering left/right angles. After that 
I shuffled the data and split 80% of data into training and 20 % into test .Finally I used
adam optimizer  so that manually training the learning rate wasn't necessary.

