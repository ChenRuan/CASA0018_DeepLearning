# Plant Health Classification using CNN

## Introduction

My project is to train a classification CNN model that can identify the health of common indoor potted plants, including orchids, aloe vera, and other plant pots, and then port the model to an Arduino Nano 33 BLE Sense board. Photographs were taken using the Arduino board with an OV7675 camera, and the model was utilized for health detection. The reason that inspired me to do this was that I noticed that many of the plants inside the CE lab were in an unhealthy state, and I wanted to be able to detect them using machine learning so that we could be alerted when a plant was deemed unhealthy, in time for those plants to be saved. The example I base this on is that I have seen databases of various plant diseases (Bhattarai, 2018), which are more readily available as machine learning solutions, so I think it is possible to use deep learning to recognize whether a plant is healthy or not.

![image](https://github.com/ChenRuan/CASA0018_DeepLearning/assets/145383140/9542fdb6-014a-4f45-be05-57fde61a674f)
![image](https://github.com/ChenRuan/CASA0018_DeepLearning/assets/145383140/c34e08af-c58b-4471-b7e7-f2c981863554)

![image](https://github.com/ChenRuan/CASA0018_DeepLearning/assets/145383140/6f4a50f9-ccb5-47a5-b497-dc4ba2793f23)

## Research Question

How to distinguish between healthy and unhealthy potted plants using CNN deep learning models? 

How to compress this image recognition model and place it on an Arduino Nano 33 BLE for practical applications?

## Application Overview

Here's a diagram of my application architecture. My input is a photo taken by an OV7675 camera with a resolution of 640*480, which then goes into an Arduino Nano 33 BLE Sense board for pre-processing, cropping and compression into 48*48 2D data for subsequent processing. After this, I used the TF Lite interpreter to run a classification deep learning model trained on Edge Impulse for recognizing the health of potted plants for processing this 2D data. This model utilized a convolutional neural network to learn a dataset containing a large number of healthy and unhealthy potted plants. After that, the output of the model was used by comparing the values of the two classifications, Healthy and Unhealthy, to determine whether the plant was healthy (healthy>0.6), unhealthy (unhealthy>0.6) or unknown (other). Finally, the results are displayed by three colors of lights on the device, green for healthy, red for unhealthy, and yellow for unknown. Users can easily get the output of the device by looking at these lit up led lights.

![image](https://github.com/ChenRuan/CASA0018_DeepLearning/assets/145383140/50088664-83c8-4bc2-accb-131f4fd22e51)


## Data

I collected most of my plant data from the web, including from plant-themed websites (e.g. Greg), Reddit (lots of unhealthy plant collected here), photographic atlases (e.g. Depositphotos), and other places. Also, I photographed the plants around me, including from labs, home, roadsides, etc. I then categorized them into healthy and unhealthy categories in my personal standard. The initial version of the dataset includes a total of about 250 photos of plants, trees, flowers, etc. in the garden, but I found that a dataset containing multiple categories of plants would only make learning more difficult, and also some plants exhibited characteristics that would make it look like it was unhealthy throughout the dataset. Therefore, I focused the plant categories into houseplants such as greenbrier, avocado potted plant, hanging orchid, aloe vera, fortune tree, spring feather, etc., expanded the dataset of photographs to 300 and cleaned up the unhealthy plant images where the unhealthy characteristics were not significant enough. After that, I uploaded these images to Edge Impulse and batch resized them to 48*48 for training the model. Here is the second version of the cleaned-up pictures.

![image](https://github.com/ChenRuan/CASA0018_DeepLearning/assets/145383140/31ff993f-8283-454d-9462-121135014161)

![image](https://github.com/ChenRuan/CASA0018_DeepLearning/assets/145383140/1ae5e6a4-2465-41ca-a89c-26de12ccb7f5)

## Model

Initially I wanted to go with the migration model on edge impulse, but I found that the migration model has very few parameters to choose from, as well as testing poorly after trying: for the first version of the dataset, the accuracy of the validation set was only about 74%, and in EON tuner, the second version of the dataset had a maximum accuracy of only 79%. Therefore, I re-chose the classification model for deep learning via convolution and pool. This model is more generalizable and I was able to tune and control it better, so it was very effective in distinguishing between healthy and unhealthy potted plants.

After continuous testing and optimization, I finally obtained a CNN model diagram that meets the expectation, as shown below. The RGB image with a resolution of 48*48 is convolved three times (one 16 filters and two 32 filters) and pooled three times (2*2), and then turned into a one-dimensional vector through the flatten layer and classified into two categories: healthy and unhealthy.

![image](https://github.com/ChenRuan/CASA0018_DeepLearning/assets/145383140/03bbea40-6700-4760-8d88-b83598c05ebc)

## Experiments

I conducted a lot of experiments to get the best parameters for the model. Initially I chose a 96*96 model for my attempts. But unfortunately, while completing the training of the model and trying to deploy it to Arduino Nano 33 BLE Sense board, I found out that there was not enough space because it only has 256KB RAM and 1MB flash memory. This situation made me realize that I need to control the size of the model including the number of image features and the complexity of the model. After this I tried using 64*64 as the image size, but increasing the complexity slightly overflowed the memory. As a last resort I chose 48*48, although it doesn't look like this had much of an impact on my model results.

With the model evaluation plugin that comes with edge impulse, I was able to access the model's performance, including accuracy and loss in the validation set, loss in the test set, and how all the data was judged, to determine the direction of changes that needed to be made. I tweaked a large number of parameters to optimize my model, including adjusting the number of convolution and pooling, the size and number of filters, and the number and parameters of Drop layers. Below I have documented the parameters that were adjusted and the associated results.

![image](https://github.com/ChenRuan/CASA0018_DeepLearning/assets/145383140/439b6ab0-08cd-4227-bf58-3fed1950da33)

So finally, comparing the accuracy and loss of the validation set, the accuracy of the test set, RAM and flash memory usage with the ability to generalize, I chose one of the better performing solutions as the final model to be deployed, which is shown below:

![image](https://github.com/ChenRuan/CASA0018_DeepLearning/assets/145383140/d5d69138-4cca-4705-9930-4e202835512d)

![image](https://github.com/ChenRuan/CASA0018_DeepLearning/assets/145383140/4086ed17-551c-4b6c-bfee-27f7f849a68d)

## Results and Observations

Even though the model seemed to turn out well, there were still some surprises when deploying it to the device. I designed the enclosure for the board on which the model was deployed to allow it to display the corresponding results via LEDs when the corresponding data is detected, as shown in the image below. Although it can achieve the desired goal, I found some obvious problems when I actually use it:

1. The model deployed on the board was more biased towards recognizing plants as unhealthy (even though the plants were perfectly healthy)
2. It takes about 3 seconds to read and transform the data, run the model and output it, which is not a good experience.
3. Even for the same plant, there are different results at different angles.

To find the source of the problem, I deployed the model on a mobile phone using the tools provided by Edge Impulse, taking advantage of the mobile phone's better arithmetic and clearer camera for comparison. As it turned out, the deployment on the phone was much better, and both problems 1 and 2 ceased to exist, but problem 3 remained. 

For problem 1, I think it might be because the OV7675 camera gets a darker image relative to the brightness of the mobile phone shot, as well as less clarity, so the shot will be green-yellowish and darker, whereas the photos in my dataset are basically all taken by mobile phones and professional cameras, and thus are more likely to be considered unhealthy after being processed by the model. I also tested lighting the plants and the error was indeed eliminated. 

![image](https://github.com/ChenRuan/CASA0018_DeepLearning/assets/145383140/a409ab4e-64de-47a6-a469-0b36b7a3a040)

Problem 2, when deployed to run locally on the phone, it can be clearly felt that the judgment speed is very fast, this is because of the Arduino Nano 33 BLE Sense board's arithmetic power is not enough to cause, but also because the complexity of my model is still high, and it needs some time to be processed. 

Problem 3 is still not solved, and I think the reason is that most of my dataset is taking pictures with a flat view angle, and few oblique or top view, so when the actual application is not detected with a flat view, there will be a large error.

![image](https://github.com/ChenRuan/CASA0018_DeepLearning/assets/145383140/fd5310b5-923f-459f-a6d8-f90d66713c9c)

Therefore, if I have more time, I think I can adjust the contrast, brightness and color gradation of the images acquired by OV7675 to make them closer to the real one, or change the camera; reduce the complexity of the model a bit while ensuring the accuracy of the model; and increase the dataset by adding pictures of potted plants taken from various angles.
