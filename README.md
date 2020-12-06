# TensorFlow 1.X Benchmarks
This repository contains various TensorFlow benchmarks. Currently, it consists of one project:

1. [scripts/tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks): The TensorFlow CNN benchmarks contain benchmarks for several convolutional neural networks.

This branch is compatible with Tensorflow 1.13, 1.14 and 1.15 (but not with Tensorflow 2.x)

## Usage

for GPU:

```
python3 ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --num_gpus=1 --xla --batch_size=64
```

for CPU:

```
python3 ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --device=cpu --data_format==NHWC --batch_size=[num cores]
```

## Results

Benchmarks meassured with this scripts are available here:

[AIME Deep Learning Benchmarks 2020](https://www.aime.info/blog/deep-learning-gpu-benchmarks-2020/)
