# 浅谈硬件加速器中的稀疏数据运算优化 - 论文笔记 II

**相关链接**
- [浅谈硬件加速器中的稀疏数据运算优化 - 论文笔记 I](https://zhuanlan.zhihu.com/p/422169740)
- [主要参考的综述文献链接](https://arxiv.org/abs/2007.00864)

## IV 概述-不同的优化角度
前文提到，针对稀疏性进行优化的硬件加速系统（Accelerator system）通常由多个部分的优化共同设计而成，有着许多的优化方法与角度，能带来不同的性能提升与硬件overhead。**本节将针对稀疏张量硬件加速器的综述文章进行概述，并在后续展开讨论**。该论文已收录在Proceedings of the IEEE，并可在arxiv上下载：[论文链接](https://arxiv.org/abs/2007.00864)。

### **1. Sparsity Encoding**
从张量的角度来看，对于稀疏张量中的非零值数据(NZ)，稀疏编码（sparsity encoding）类似一种压缩（Compressed）编码的功能，将原张量中的零值忽略，从而减少数据规模。从硬件设计角度来看，在从片外存储器传输至硬件加速器的过程中，Sparsity Encoding可以减少带宽和片上缓存的需求。学术界提出了各种各样的编码方式，后续展开讨论。

### **2. Data Extraction**
Data Extraction可以看作是Sparsity Encoding的逆过程。加速器从片外存储中获得Compressed Tensor数据后，需要将其展开成原始数据格式送往PE模块进行高效运算。简而言之，Data Extraction模块需要从压缩数据中获取所有NZ的value（数值），index（在张量中的位置）与match（例如，NZ input activations, NZ weights和partial summation这三者对应位置的匹配），从而PE只需要完成这些NZ的运算，跳过零值。

### **3. Memory Management**
如论文笔记I所述，探索稀疏加速器中的数据复用是十分必要的。而为了支持高效的数据复用，首先要进行片上缓存的设计。除此以外，如何设计缓存地址的**控制器**也是一个难点。例如，由于数据的复用性，可能会出现多个计算单元同时访问同一个数据的情况，控制器要规避这种访存冲突；同时，数据稀疏性导致控制器需要尽可能减少硬件的miss penalty，即尽可能“跳读”存储器中不会参与当前运算的值。

### **4. Communication Networks (System Architecture)**
除了memory架构支持以外，灵活的系统互连结构也对数据复用带来很大的帮助。数据流（dataflow）是硬件加速器中讨论的重点，Memory如何将输入数据送往每个PE，每个PE如何选择将算完的部分和（Partial Sum）传回给总线或其他的PE（例如systolic array，需要将partial sum送往相邻的PE进行累加），以及最终如何将数据收集写回片外存储器，每一种不同的数据分发方式都对应着一种不同的数据流，而不同的数据流会对latency和energy efficiency产生很大的影响。这一部分我会在接下来的讨论中结合其他一些讨论数据流的论文，重点分析。

### **5. PE Architecture**
注意区分PE Architecture和System Architecture之区别。PE Architecture主要讨论PE内部的电路优化细节，如MAC运算的具体实现、可重构的数据精度和PE内部的计算数据流等；结合第二点Data Extraction来说，有的工作会讨论在数据传输端进行NZ Extraction，有的则会在PE内部进行数据扩展，找到NZ数据。

### **6. Load Balancing**
我认为在稀疏优化的硬件加速器设计中，Load Balancing是最重要也是最有挑战性的优化点之一。由于数据稀疏性，每个PE输入的tensor规模和需要完成的MACs数量都可能不同。workload较小的PE率先完成之后，显然不能低效地等待其他workload较大的PE，否则会造成PE的idle从而降低处理效率。如何**动态平衡**每个PE之间的load，成为硬件设计者的一个挑战。

### **7. Write-Back and Postprocessing**
除了PE之间的系统数据流、PE内部的计算数据流以外，稀疏张量的运算（以深度神经网络DNN举例）当然不仅仅是MAC运算，还有后续的batch normalization，非线性激活函数，残差链接（ResNet）等。如何将PE阵列计算得到的MAC结果送往这些模块进行处理（是否要进行稀疏编码，是否要同步处理）以及写回片外存储器，也是硬件加速器设计的一个优化点。

### **8. Compilation Support**
现阶段学术界针对稀疏优化的编译器少之又少，大多基于regular tensor（即dense tensor）运算来进行设计。然而我只是一个刚开始接触编译器的小白，所以只能边学边总结一些编译器优化的例子和opportunities

## V 编码（Encoding）
一个稀疏张量可以被压缩编码，从而跳过零值，减少数据量。通常来说一个编码后的tensor包含实际值(data，即原tensor中的NZ)和元值（metadata，即原tensor中NZ的位置信息）。加速器得到metadata信息后，可以通过Data Extraction来获取NZ之间的匹配信息，从而进行NZ的IA×W+PSUM（输入值×权重+部分和），跳过零值计算。下面将简略介绍各种常见的编码方案。

### **常见的编码方案**

#### **1) COO**
Coordinate (COO) 编码中，metadata存储了NZ的绝对位置（即其在原tensor中的坐标）
#### **2) RLC**

#### **3) Bitmap**

#### **4) CSR/CSC**

#### **5) CSF**

#### **6) Structured Sparsity**
