# Edgerun FaaS Optimizations

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)


This repository contains various optimizations for resource management techniques.
Leveraging the [faas project](https://github.com/edgerun/faas) makes them ready-to-use for real-world clusters and
simulations at the same time.
This project collects different strategies for solutions around: scheduling, scaling and load balancing.
It is meant to serve as a central ("market")place to implement and provide novel resource management techniques.
The main goal is to provide mix & match components to conveniently evaluate different strategies.

Therefore, this project will contain only bare-minimum execution/test code.
This project is intended to be used as dependency when performing sophisticated real-world evaluations or simulations.

Install
=====

To install all dependencies, run:

    make venv


Run
===

**TODO**

Project Structure
=================

All code is contained in the `faasopts` module.
Currently, we focus on the three main resource management techniques: autoscaling, scheduling and load balancing.
You can find implemented approaches in the respective module, but we provide information for each in the following.

Autoscalers
===========

This autoscaler replicates the behavior of the [official Kubernetes Horizontal Pod Autoscaler](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/).
Originally, HPA only supported CPU-based scaling but has been extended to include various custom metrics.
To this end, we implement different variants of this autoscalers.
For example, we provide a HPA implementation that is CPU-based (`HCPA`) and one that takes the latency as metric (`HLPA`).

You can find the base `HPA` class and both implementations in the `faasopts.autoscalers.k8s.hpa.global` module.

Schedulers
==========

**TODO**

Load balancing
==============

**TODO**