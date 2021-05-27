# Eye Closure State Detection using CNN

This repository built a simple CNN structure based on LeNet-5 to detect eye closure state. Given a facial image, the eye region is extracted through Dlib library and fed into the network for classification of open state or closed state.

![](https://github.com/zeyuchen-kevin/eye_state/raw/master/Images/eye_extraction.png)

**eye region extraction using Dlib library**

## Result

Using learning rate of 0.001, weight decay of 0.001, batch size of 32 and running 50 epochs:

![](https://github.com/zeyuchen-kevin/eye_state/raw/master/training_plot.jpg)

