# MLops Optimization

This repo serves as a test to see how we can improve upon ML serving latency. Massive thanks to [comet ml](!https://github.com/comet-ml/blog-serving-hugging-face-models), which this is based off of.

## Version 1 - Basic Attempt

Version 1 is a first attempt at building a model; using BERT to make a precision on some sample text. This is what an initial attempt at creating a Dockerfile would be like.

Docker commands:

1. `docker build -t v1 version_1/` 
2. `docker image ls` 
3. `docker system df -v`
4. `docker container ls -s`

We can see here then that the image is 3.78GB, NOT including downloading the model. After starting the container, that goes up to 4.22GB (since you have to download BERT from transformers).

## Version 2 - Better Software Practices

Version 2 involves switching to lightweight versions of torch (using torch cpu), as well as using a slimmer version of python. This brings the image size down to 1.12GB, and starting the container up to 1.56GB. This should be considered the bare minimum that needs to be done; here, versions are pinned as well.

The same docker commands from before are used, just switched out to use `v2`. 

## Version 3 - Distillation and Onnx

Version 3 introduces some new enhancements made by switching to a model using distillation, i.e. in this case [DistilBERT](!https://arxiv.org/abs/1910.01108), which is 40% smaller. In addition, we can use onnx to help reduce model size. The image itself becomes larger, at 1.72GB, however that is due to the fact that we now have both the original distilbert model + the onnx version. 

## Version 4 - Multistage Docker Builds

In this version, the Docker image shrinks by doing a multi-step build and only copying over the relevant items needed (as well as copying the tokenizer locally, so as to save time on container start). Now we're down to 1.25GB for the image size, and on container start, the size is still the same.


| Version  | Image Size | Container Size    |
| :---     |    :----:   |          ---: |
| V1*  | 3.783GB        |   4.22GB    |
| V2*   | 1.051GB        |  1.61GB     |
| V3   | 1.602GB        |  1.72GB     |
| V4   | 1.125GB        |  1.125GB     |

*Note for V1 and V2, the model isn't downloaded yet (will be done at container start time), which is why you see the gap

## ToDOS

1. Quantization
2. Switching to Go vs Python for serving

* Note - something that potentially could be done to help shrink the size is to actually go into torch and start pruning tests. Yes, prune some tests - turns out a good chunk of the pytorch size comes from test modules! Obviously very dangerous, but we could shrink ~ 30% of the torch download size on disk. We would need some place to zip up and package the environment manually, vs downloading everything via `pip` in the dockerfile.