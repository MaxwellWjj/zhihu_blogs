# Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows: A Data-Centric Approach Using MAESTRO

## Paper Information
Hyoukjun Kwon, Prasanth Chatarasi, Michael Pellauer, Angshuman Parashar, Vivek Sarkar, and Tushar Krishna. 2019. Understanding Reuse, Performance, and Hardware Cost of DNN Dataflow: A Data-Centric Approach. In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture **(MICRO '52)**. Association for Computing Machinery, New York, NY, USA, 754–768. DOI:https://doi.org/10.1145/3352460.3358252


## Objective
The performance and energy efficiency of DNN accelerators depend on (1) target DNN model and its layers types/dimensions, (2) dataflow, and (3) available hardware resources and their connectivity. These three dimensions are tightly coupled, and optimizing DNN accelerators across these dimensions is a challenging task. This paper shows a data-centric angle for optimize DNN accelerator, co-optimizing the hardware microarchitecture and the dataflows.

## Background
1) Abstract DNN ccelerator architecture. The illustrated base architecture can be hierarchically organized.

![Image](https://pic4.zhimg.com/80/v2-d1160f89e7cf7838fd4038bcfd8e60a4.png)

2) Data reuse in DNN accelerators - reduction and multicast.

![Image](https://pic4.zhimg.com/80/v2-e49601ca4973048112575e3d5fc41ce6.png)

3) **Traditional dataflow representation - represented as loop-nest**. The figure uses an example 1D convolution.

![Image](https://pic4.zhimg.com/80/v2-31b74c5ff5c6023a8c0b1b605185cacf.png)

Usually, we use *stationary-based* statements to define the dataflow. For instance, weight stationary, output stationary and input stationary.

**However, this loop-nest represenation is difficult for hardware designers to find the data reuse opportunities, and then cannot develop an accurate evaluation model/framework to get the performance and cost with the selected dataflow and microarchitecture.**

## Methods
1. Data-centric representation. This work describes DNN dataflow as data-centric directives. The three key words are *spatialmap*, *temporalmap* and *cluster*.

    1.1 SpatialMap: specifies a distribution of a dimension across PEs.
    
    1.2 TemporalMap: specifies a distribution of a dimension across time steps in a PE.
    
    1.3 Cluster: supports the simultaneous spatial distribution of multiple data dimensions

Illustration figure:

![Image](https://pic4.zhimg.com/80/v2-639acdef05100000c51c96f439ea209f.png)

Say, data mappings related to a map in the outer position get updated after a full exploration of a map in the inner position. This inherent assumption can limit certain dataflow behaviors where one might be interested in simultaneously exploiting spatial distribution of more than one data dimensions. Therefore, we need cluster to fix this problem.

2. Hardware Implementation Choices for supporting spatial and temporal reuse. For spatial and temporal reuse, they both have multicast and reduction implementaion in hardware design, as this figure presents.

![Image](https://pic4.zhimg.com/80/v2-cfe8ab5919476f3901a291e94727e829.png)

3. After getting the hardware architecuture and DNN dataflow, this work develop an framework (MAESTRO) to quantitatively estimating runtime and energy efficiency of dataflows on a target DNN model and hardware configuration. The report results of MAESTRO include *performance* and *cost*. The most important results are **latency** and **energy consumption**.

![Image](https://pic4.zhimg.com/80/v2-3998becbc33285837f6039c30eb7bb04.png)

## Evaluation
1. Case Study I. The authors select some typical accelerators in academic world, analyze the corresponding dataflow and report the results by MAESTRO framework. It can be concluded that for different DNN model, even different DNN layers, the performance and cost varies between different hardware / dataflow.

![Image](https://pic4.zhimg.com/80/v2-19332146f9eded21602d85e63f68ca8d.png)

![Image](https://pic4.zhimg.com/80/v2-e371a836b72130277a65de70d847dfc2.png)

2. Case Study II - Design Space Exploration (DSE). The DSE tool sweeps a target design space specified in the range of each parameter and search granularity. However, it skips design spaces at each iteration of hardware parameters by checking the minimum area and power of all the possible design points from
inner loops of hardware parameters. This framewowk allows it to skip invalid design points in a various granularity that reduces a large number of futile searches, which led to a large effective DSE rate ranging from 3.3K to 0.46M designs per second.

![Image](https://pic4.zhimg.com/80/v2-c8a6bd79a951911d092f3dbb773ac273.png)

## Summary
This work aims to co-optimize DNN accelerator microarchitecture and dataflow to achieve both **low latency** and **energy efficiency**. To solve this problem, this work presents data-centric directives to describe DNN dataflows. Based on this data-centric presentation, all reuse opportunities can be easily obtained and presented. With the help of the data-centric directives, traditional 7-dimension for-loops dataflow can be translated to a **Data-centric Representation**, in which the dataflow become plainer for hardware mapping. In general, the data reuse can be categorized as *spatial reuse* and *temporal reuse*, and the reuse can be realized by *multicast* or *reduction* in hardware design. 

Futhermore, the authors develop an analysis and evaluation framework called MAESTRO to search the best dataflow (i.e. low latency and high energy efficiency). The input of this framework is hardware description, DNN dataflow represented by data-centric directives and DNN model, while the output is the performance report (latency, BW etc.) and cost (energy, buffer size etc.). We can further use these analysis results for selection of dataflow and hardware architecture.

## Notes / Highlights
It is my first paper notes in HKU. I believe I will try my best to insist it.