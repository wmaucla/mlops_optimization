# MLops Optimization

This repo serves as a test to see how we can improve upon ML serving latency. Massive thanks to [comet ml](!https://github.com/comet-ml/blog-serving-hugging-face-models), which this is based off of.

## Version 1

Version 1 is a first attempt at building a model; using BERT to make a precision on some sample text.

Docker commands:

1. `docker build -t v1 version_1/` 
2. `docker image ls` 

We can see here then that the image is 3.78GB, NOT including downloading the model. After starting the container, that goes up to 4.22GB (since you have to download BERT from transformers).

Doing 10 runs, on average takes around 9 seconds

## Version 2

Version 2 involves switching to lightweight versions of torch (using torch cpu), as well as using a slimmer version of python. This brings the image size down to 1.12GB, and starting the container up to 1.56GB.

The same docker commands from before are used, just switched out. 