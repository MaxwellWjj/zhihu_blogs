#! https://zhuanlan.zhihu.com/p/422169740
# 浅谈硬件加速器中的稀疏数据运算优化 - 论文笔记 I

## I 概述
近年来硬件加速器随着AI热潮而成为学术/工业界的热点之一（图1描述了硬件加速器的抽象示意图，引用自[5]），其主要目的是为了追求在低延迟、高能效下完成AI模型的训练与推断（例如，边缘计算设备对能耗的要求以及高精度场景中对延迟的要求）。在每年的计算机体系结构的顶会中（ISCA, MICRO, HPCA, ASPLOS etc.），机器学期模型硬件加速相关的工作占据了很大的比重。近年的工作中，对于硬件加速器的优化有着各种各样的角度，举几个学术界热门的工作来说，TPU[1]从处理单元阵列（PE Array）层次出发，将Systolic Array架构及其数据流引入DNN加速器中；Eyeriss[2]同样对数据流进行优化，提出了Row Stationary数据流以提升能效；而在电路层次的优化中，打破冯-诺依曼结构的存内计算（In-memory Computing, [3]）无疑是最具代表性的方向之一；除此之外，还有许多学者从编译器角度出发，针对不同的硬件target进行优化[4]。

![Image](https://pic4.zhimg.com/80/v2-27df68e77e9693cad5306d7e8fe11c3b.png)
<center style="font-size:14px;color:#C0C0C0">图1 硬件加速器的抽象示意图，引用自[5]</center> 

随着机器学习模型参数规模的快速增长（如图2，引用自[5]），许多硬件加速相关的工作对于如何利用机器学习模型的稀疏性（Sparsity）展开研究，这也是笔者近期研究的方向之一。以DNN举例，activations和weights张量（Tensor）中存在大量的零值或近零值数据，这些数据参与的MAC运算（Multiply-Accumulation）对于最终结果没有任何贡献，因此如何高效地跳过这些运算成为了硬件加速器的关键优化点，也存在许多挑战。这几篇论文笔记是关于笔者近期学习的综述文献[5]，其归纳总结了近年来典型的稀疏数据优化相关硬件加速器的工作，该文章已收录在Proceedings of the IEEE，并可在arxiv上下载：[论文链接](https://arxiv.org/abs/2007.00864)。

![Image](https://pic4.zhimg.com/80/v2-358264b4df58bcd1c75342fdc11eecb3.png)
<center style="font-size:14px;color:#C0C0C0">图2 DNN模型参数规模增长，引用自[5]</center> 

## II 模型中的张量稀疏性与非规整性
以深度神经网络为代表的机器学习模型中存在大量的数据稀疏性，图3清晰明了地总结了DNN模型中可能存在的三种典型稀疏性[6]。

![image](https://pic4.zhimg.com/80/v2-460487ce93801788d2b074aaa5ad9a7b.png)
<center style="font-size:14px;color:#C0C0C0">图3 DNN模型中的三种稀疏性，引用自[6]</center> 

其中，input sparsity即input activations tensor中存在上一层激活函数产生的零值（如ReLu），其对应的MAC运算可被忽略；weight sparsity即由于算法模型中的pruning，quantization优化而产生的零值或近零值，均可在硬件推断中忽略；output sparsity则是一种预测机制，即在input activation和weight均非零的情况下，预测可能产生的输出零值，则在硬件加速器中，这些MAC运算也可以跳过，直接将对应的输出置零。在稀疏性优化的帮助下，DNN的参数张量规模可以压缩到很小的比例。举例来说，ADMM-NN [7] 利用 weight pruning 和 quantization, 减少了 AlexNet, VGG-16 和 ResNet-50 模型的参数规模分别达到了  99×, 66.5× 和 25.3× (仅造成最多 0.2% accuracy loss)。

然而这些稀疏张量在硬件加速器中的优化（即跳过计算）并非易事。**核心问题在于，稀疏性的引入使得硬件处理中的张量变得不规整（irregular）**，学术界和工业界的大部分加速器设计的target均为regular张量处理。

>文献[5]原文：They are designed for **performing structured computations with regular memory accesses and communication patterns**. Without special support for sparse tensors, they fetch all the data, including zero values from memory, and feed into PEs, thereby wasting the execution time. Sparsity, especially unstructured, **induces irregularity in processing since nonzeros (NZs) or blocks of NZs are scattered across tensors**.

因此，利用机器学习模型中张量稀疏性的目标是大大减少零值的计算（只对非零值 - Nonzeros, NZs 进行运算）、通信和存储，而在保持性能提升的情况下，避免增加过多的功耗和面积开销。这其中，软硬件协同优化将成为关键点之一。

## III 稀疏性优化的加速器系统
一个硬件加速系统（Accelerator system）通常由多个部分的优化共同设计而成（如图4所示，引用自[5]）。首先，可以针对稀疏性，量化以及模型压缩，设计适合硬件部署的算法模型，从而让硬件加速器能够以较小额外开销来处理这些稀疏张量；其次，一个硬件系统离不开编译器（compiler）支持，通常来说，硬件加速领域的编译器需要将算法处理模型转化成中间表达（IR），并根据不同的终端硬件架构，生成不同的数据映射、数据流，并最终生成对应的执行代码（code generation）；对于稀疏张量，设计者还需要考虑如何对稀疏数据进行编码（coding）和量化，以方便在加速器的片上缓存中存储与后续的处理；引入稀疏性后，还需要考虑如何设计片上与片外的数据交互，以最大程度复用片上的数据，减少off-chip access；在硬件架构上，优化点集中在如何选择高效的数据流以及任务分配（Work Allocation）来处理这些稀疏张量计算，以及如何设计不同PE间的互连来支持这样的数据流。

![Image](https://pic4.zhimg.com/80/v2-a0d57f2d76209e03184952c60ff09af9.png)
<center style="font-size:14px;color:#C0C0C0">图4 利用稀疏性进行加速的硬件系统示意图</center> 

其中，我认为最有挑战性的优化点是：
1. 如何在高度稀疏性的模型中提高数据复用，以减少片外通信的带宽需求。在不考虑模型中张量稀疏性的情况下，数据复用在以DNN为代表的机器学习模型中非常常见。数据复用通常可分为IA reuse(同个输入activation共享给多个weights进行运算，以下以此类推)，weights reuse和partial summation reuse。然而在引入稀疏性的情况下，由于包含0值的计算会被跳过，从而导致总运算量MACs减少，这三种复用都会显著降低。文献[5]做了关于数据复用性降低的图表，如图5所示，可以看出sparse tensor和dense tensor的计算对比下，数据复用性（用总运算量MACs/总数据量data来表达）甚至会出现小于1的情况（例如对于MLP）。复用性的降低会提升片外通信带宽需求，通俗来说，我们需要更快的数据传输速度，才能支持片上更快的处理速度（因为MACs的减少），若设计者能在MACs减少的情况下依然尽可能保持数据的复用性，来减少data的传输量，那么片外通信的总线带宽压力便可以减小。

> 文献[5]原文 - For processing high sparsity (e.g., 90%+) and low reuse, it becomes challenging to engage functional units into effectual computations. This is because, with low arithmetic intensity, **required data may not be prefetched at available bandwidth.**

![Image](https://pic4.zhimg.com/80/v2-02d1c8a02498f46f36d2385f3139a60c.png)
<center style="font-size:14px;color:#C0C0C0">图5 引入稀疏张量后，DNN模型复用降低</center> 

2. 如何解决irregular tensor带来的imbalance问题。这个挑战非常容易理解，因为稀疏性的引入，每个PE输入的tensor规模和需要完成的MACs数量都可能不同。workload较小的PE率先完成之后，显然不能低效地等待其他workload较大的PE，否则会造成PE的idle从而降低处理效率。在这种情况下，许多工作提出了不同的方式来解决这个问题，例如每个PE分配一个FIFO，存储多个workload，来避免idle状态，或使用一个overhead很大的调度控制器，来将其他PE的workload分配给率先完成的PE。如何衡量这些balance方案，选取额外开销最小的最优解，甚至对架构进行可重构配置来支持balance方案，成为了一个很大的挑战。

![Image](https://pic4.zhimg.com/80/v2-00555e5d7165738bb8b14b34e3647e67.png)
<center style="font-size:14px;color:#C0C0C0">图6 PE workload imbalance示意</center> 

3. 针对稀疏张量计算的编译器。如前所述，一个完备的硬件加速器系统离不开编译器的支持，然而现阶段学术界针对稀疏优化的编译器少之又少，大多基于regular tensor（即dense tensor）运算来进行设计。而这其中讨论空间最大的我认为是针对稀疏性来设计特殊的IR，从而能对不同硬件平台进行compiler optimization与code generation。例如，使用Polyhedral representation 还是 Nonpolyhedral representation，各自有什么样的优缺点。

## IV 后续
其实这篇综述我以及学习完了，但一直找不到时间整理成中文博客，写文章实在是太花时间了……我会尽力将本综述后续的章节博客完成，并加入一些其他文献的分析和我的理解。

后续的章节会对III中描述的系统中的每个部分进行展开论述。

## V 参考文献
[1] N. P. Jouppi *et al.*, “In-datacenter performance analysis of a tensor processing unit,” in *Proc. ACM/IEEE 44th Annu. Int. Symp. Comput. Archit. (ISCA)*, Jun. 2017, pp. 1–12.

[2] Y.-H. Chen, T. Krishna, J. S. Emer, and V. Sze, “Eyeriss: An energy-efficient reconfigurable accelerator for deep convolutional neural networks,” *IEEE J. Solid-State Circuits*, vol. 52, no. 1, pp. 127–138, Jan. 2017.

[3] Wang, Y., Tang, H., Xie, Y. *et al.* An in-memory computing architecture based on two-dimensional semiconductors for multiply-accumulate operations. *Nat Commun* 12, 3347 (2021).

[4] T. Chen *et al.*, “TVM: An automated end-to-end optimizing compiler for deep learning,” in *Proc. 13th USENIX Symp. Oper. Syst. Design Implement. (OSDI)*, 2018, pp. 1–17.

[5] S. Dave, R. Baghdadi, T. Nowatzki, S. Avancha, A. Shrivastava and B. Li, "Hardware Acceleration of Sparse and Irregular Tensor Computations of ML Models: A Survey and Insights," in *Proceedings of the IEEE*, vol. 109, no. 10, pp. 1706-1752, Oct. 2021.

[6] J. Lee, S. Kang, J. Lee, D. Shin, D. Han and H. -J. Yoo, "The Hardware and Algorithm Co-Design for Energy-Efficient DNN Processor on Edge/Mobile Devices," in *IEEE Transactions on Circuits and Systems I: Regular Papers*, vol. 67, no. 10, pp. 3458-3470, Oct. 2020.