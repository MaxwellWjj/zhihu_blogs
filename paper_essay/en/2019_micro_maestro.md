# Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows: A Data-Centric Approach Using MAESTRO

## Summary
This work aims to co-optimize DNN accelerator microarchitecture and dataflow to achieve both **low latency** and **energy efficiency**. To solve this problem, this work presents data-centric directives to describe DNN dataflows. Based on this data-centric presentation, all reuse opportunities can be easily obtained and presented. With the help of the data-centric directives, traditional 7-dimension for-loops dataflow can be translated to a **Data-centric Representation**, in which the dataflow become plainer for hardware mapping. In general, the data reuse can be categorized as *spatial reuse* and *temporal reuse*, and the reuse can be realized by *multicast* or *reduction* in hardware design. 

Futhermore, the authors develop an analysis and evaluation framework called MAESTRO to search the best dataflow (i.e. low latency and high energy efficiency). The input of this framework is hardware description, DNN dataflow represented by data-centric directives and DNN model, while the output is the performance report (latency, BW etc.) and cost (energy, buffer size etc.). We can further use these analysis results for selection of dataflow and hardware architecture.

## Research Objective
What is the objective?

## Background / Problem Statement
What problem has been solved?

## Method(s)
Algorithm or methodology

## Evaluation
Put some data

## Conclusion
make some conclusion

## Notes / Highlights


## Important References
Some important ref (mostly the background and benchmark works)