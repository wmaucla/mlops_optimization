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

The same docker commands from before are used, just switched out to use `v2`. 

## Version 3

Version 3 introduces some new enhancements made by switching to a model using distillation, i.e. in this case [DistilBERT](!https://arxiv.org/abs/1910.01108), which is 40% smaller. In addition, we can use onnx to help reduce model size.

* Note - something that potentially could be done to help shrink the size is to actually go into torch and start pruning tests. Yes, prune some tests - turns out a good chunk of the pytorch size comes from test modules! Obviously very dangerous, but we could shrink ~ 30% of the torch download size on disk. We would need some place to zip up and package the environment manually, vs downloading everything via `pip` in the dockerfile.

## ToDOS

1. Quantization
2. Switching to Go vs Python for serving