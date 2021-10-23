# 浅谈硬件加速器中的稀疏数据运算优化 - 论文笔记 II

**相关链接**
- [浅谈硬件加速器中的稀疏数据运算优化 - 论文笔记 I](https://zhuanlan.zhihu.com/p/422169740)
- [主要参考的综述文献链接](https://arxiv.org/abs/2007.00864)

## I 概述-不同的优化角度
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

## II 编码（Encoding）
一个稀疏张量可以被压缩编码，从而跳过零值，减少数据量。通常来说一个编码后的tensor包含实际值(data，即原tensor中的NZ)和元值（metadata，即原tensor中NZ的位置信息）。加速器得到metadata信息后，可以通过Data Extraction来获取NZ之间的匹配信息，从而进行NZ的IA×W+PSUM（输入值×权重+部分和），跳过零值计算。下面将简略介绍各种常见的编码方案，我将会把学术语言尽量用通俗易懂的方式表达。

### **常见的编码方案**
图1列举了学术界常见的一些sparse tensor编码方案。图1(a)是原sparse tensor的值，在这里取tensor的维度为2，即一个矩阵，其中有4个NZ和5个零值。

![图1 sparse tensor编码方案示意图，引用自[1]](https://pic4.zhimg.com/80/v2-b5c88e4e80d0ecf903bc06a54c81b117.png)

#### **1) COO**
Coordinate (COO) 编码中，metadata存储了NZ的绝对位置（即其在原tensor中的坐标）,加速器的运算单元可以直接获取其位置信息而不需要进一步的解码。图1(b)展示了COO编码的例子，使用了两个维度的index来存储metadata，可见其效率不高。根据论文分析，假设NZ的数量为NNZ (Number of NZ)，则hardware storage overhead可以由下式近似计算得到，其中n为dimensions，$d_{i}$ 为每一维的长度，log2函数用来转化为“所需要的bit位宽”，下同。

$$NNZ \times \sum_{1}^{n}\left\lceil\log_{2} d_{i}\right\rceil$$

为了节省COO编码的metadata存储开销，有工作提出将位置坐标展开（flatten）成一维的位置信息，如图1(c)所示。这种编码方式在tensor维度较低，规模较小的情况下能够节约很大的空间，但倘若tensor规模较大，则其一维展开之后的metadata需要较宽的寄存器来存储，会造成更大的开销（注意，开销按bit计算）。这种编码方式下的storage overhead可以表示为：

$$
N N Z \times\left\lceil\log _{2} \prod_{1}^{n} d_{i}\right\rceil
$$

#### **2) RLC**
RLC编码的全称为Run-Length Coding。简单表述这种编码方式为：metadata储存的值为flatten操作后的相邻两个NZ之间的距离（即中间有多少个零值）。如图1(d)所示，2和3之间隔了一个零值，那么metadata的第二个数为1，3和5之间在flatten操作后是相邻的，因此对应的第三个数位0，以此类推。笔者认为这种方案适合“串行读取&处理”的方案，例如PE需要串行地将2，3，5，7依次取出，参与MAC运算，因为地址生成器可以根据metadata的值直接在当前地址上累加。RLC编码的storage overhead取决于两个非零值之间可能出现的最长距离，因此不好直接给出表达式。对于大规模且稀疏度非常高的tensor（两个NZ相隔非常远），RLC编码方式需要进行改进，否则将占用很大的开销。

#### **3) Bitmap**
Bitmap编码中的metadata会存储原tensor中的每一个element是否为0的信息，每个element对应1bit，若该element为NZ，则metadata中的这一bit为1，否则为0。图1(e)展示了这种编码策略，注意bitmap的长度是原tensor的总element数量，因此这种方法适合稀疏度较低的tensor。当稀疏程度较高时，采用bitmap明显会造成很大的浪费（因为大部分metadata bits均为0）。Bitmap编码的存储开销可以表示为下式（因为bitmap的每个metadata信息均为1bit）。

$$
\prod_{1}^{n} d_{i}
$$

#### **4) CSR/CSC**
Compressed Sparse Row (CSR) 和 Compressed Sparse Column (CSC)均属于compressed sparse系列编码。CSR将sparse tensor中每一行的NZ压缩到一起（即row-wise压缩），再进行类似COO的编码，即把图1(a)转化为((2,3,-),(5,-,-),(7,-,-))这样的残缺的矩阵。

在这种压缩情况下，不再需要图1(b)中的xy两个coordinate来表达位置，而是采用ptr-idx的方式，如图1(f)，ptr用来表示每一行的NZ数量（ptr[i + 1] - ptr[i]），例如图1(a)有3行，那么ptr的长度为4，ptr[1]-ptr[0]=2，表示第一行有2和3两个NZ，以此类推，再用行内的idx来表示NZ位于该行中的第几列，如3对应的idx为2。其storage overhead可表示为（注意我们默认tensor只有两个维度，即已经通过tensor unfold转化成了二维的矩阵，只有行、列信息，更多关于tensor unfold操作可以参考[2]）。

$$
N N Z \times\left\lceil\log _{2} d_{1}\right\rceil+\left(d_{0}+1\right) \times\left\lfloor\log _{2} N N Z+1\right]
$$

Compressed Sparse Column (CSC)与CSR类似，只不过改成了按行压缩，即矩阵变为了((2,-,3),(5,-,7),(-,-,-))，idx表示每一列内的NZ所在行数。其storage overhead可表示为：

$$
N N Z \times\left\lceil\log _{2} d_{0}\right\rceil+\left(d_{1}+1\right) \times\left\lfloor\log _{2} N N Z+1\right\rfloor
$$

#### **6) Structured Sparsity**
在sparse tensor的引用中，除了随机的sparsity以外，还有一些场景中稀疏性表现出structured形式，如图2所示。(a)即为最general的随机sparsity，没有任何规律性，只能采用前几种编码方案。但(b),(c),(d)均展现了一些特殊的，有规律的sparsity。对于这些sparse tensor的优化，metadata可以非常简单地用很小的overhead实现。例如(b)，我们只需要1 bit的metadata，用来表示每个block（4个elements）是否全部为NZ。当然，这种Structured Sparsity在应用中实现，一般需要对数据进行预处理，例如调换数据存储中的顺序。

![图2 不同的sparsity pattern。改图引用自[1]](https://pic4.zhimg.com/80/v2-b29f9f305f5a8bf56e4246325af50fc4.png)

#### **7) 不同编码方式的选择**
图3给出了对于相同规模tensor，不同sparsity程度(百分比越大表示越稀疏，NZ越少)情况下，不同编码方案的storage saving，即引入稀疏编码后的存储开销/不考虑稀疏性而采用dense tensor存储方案的开销。不过我的愚见是，根据之前的几个公式，影响storage saving的因素不仅有sparsity程度，还有tensor的规模，所以在选择编码方式的适合还应该进行更全面的评估。**对于不同的应用场景和稀疏程度，选择特定的编码方式是非常有必要的优化**。

![图3 对于相同规模tensor，不同sparsity程度情况下，不同编码方案的storage saving。改图引用自[1]](https://pic4.zhimg.com/80/v2-1753f1c07318e5556fd5783276165791.png)

## III 解码与非零值的匹配
如Section I所述，Data Extraction可以看作是Sparsity Encoding的逆过程。加速器从片外存储中获得Compressed Tensor数据后，需要将其展开成原始数据格式送往PE模块进行高效运算，这其中涉及到NZ的匹配问题。以深度神经网络（DNN）举例，我们可以用for循环来表达其tensor运算过程：

$$
\begin{aligned}
&for(\mathbf{n}=0 ; \mathbf{n}<N ; \mathbf{n}++) \\
&for(\mathbf{k}=0 ; \mathbf{k}<K ; \mathbf{k}++) \\
&for(\mathbf{c}=0 ; \mathbf{c}<C ; \mathbf{c}++) \\
&for(\mathbf{y}=0 ; \mathbf{y}<Y ; \mathbf{y}++) \\
&for(\mathbf{x}=0 ; \mathbf{x}<X ; \mathbf{x}++) \\
&for(\mathbf{r}=0 ; \mathbf{r}<R ; \mathbf{r}++) \\
&for(\mathbf{s}=0 ; \mathbf{s}<S ; \mathbf{s}++) \\
&\mathrm{O}[\mathbf{k}][\mathbf{y}-\mathbf{r}][\mathbf{x}-\mathbf{s}]+=\mathrm{W}[\mathbf{k}][\mathbf{c}][\mathbf{r}][\mathbf{s}] \times \mathrm{I}[\mathbf{c}][\mathbf{y}][\mathbf{x}]
\end{aligned}
$$

for-loop表示DNN的方法会在之后有关dataflow的章节再次用上并详细讨论，可以参考MAESTRO这篇工作[3]。可以看出，DNN需要完成的MAC运算对于数据的index有严格的匹配要求，因此PE需要从得到的稀疏编码数据以及metadata信息中，得到原始dense tensor中的index对应关系，完成数据的匹配。总的来说学术界现有的extraction方案可以分为3种（注意，以下均以DNN运算举例）。

### **1) 查找**
顾名思义，在这种情况中，PE需要做的就是将metadata转化为NZ在dense tensor中的真实位置，然后根据该位置（index）直接查找到另一个tensor（**通常必须要是dense tensor**）中对应的element，取出并运算，如图4和5所示。图4的例子是input activation参与了稀疏编码，但weights没有参与，所以通过input activation的metadata恢复出index后，查找weights (filter) tensor中的值而参与运算。而图5则相反，input activation是dense的，weights参与了稀疏编码。当然，也可以两个tensor都是sparse的，只不过查找逻辑要更复杂（因为两个稀疏编码后的tensor都需要查找原dense tensor中的index，但编码方式不一定相同，可能要两个查找表……），开销更大。

![图4，引用自[1]](https://pic4.zhimg.com/80/v2-b15a3439084561f4851d96a8ed715f43.png)

![图5，引用自[1]](https://pic4.zhimg.com/80/v2-19f9d96e554f4af734bfa77ec57f4a1a.png)

用查找的方案来进行extraction，局限性较大，尤其是对于高度稀疏的场景，需要很大的multiplexer来查找dense tensor中的位置。因此许多工作采用了其他的优化方案。

### **2) 比较**
比较方案，就是对于查找方案的改进。PE不需要通过metadata恢复出原dense tensor的index再去查找，而是直接比较两个sparse tensor的metadata从而完成数据匹配。典型的例子是SNAP [4]，采用了一个比较器阵列直接将input activation的metadata与weights的metadata进行比较，然后直接输出匹配的pairs进行运算，如图6所示，比较器结果为1表示这两个metadata是匹配的。

![图6，引用自[1]](https://pic4.zhimg.com/80/v2-595a2c365349d615da9e7e4b5575a660.png)

当然，这种方案可以在一个周期内完成匹配，代价是硬件开销很大。若加速器对于latency没有很高的要求，可以采用Eyeriss V2中提到的单个element匹配整个vector的方式，然后形成element之间的pipeline。

### **3) 直接运算**
这种方式仅限于structured sparsity，即sectionII中提到的第六点。对于这种规律性的稀疏，加速器可以直接“告知”PE哪些位置的NZ是匹配的，从而大大减少硬件开销。同样地，这种方式一般需要对数据进行预处理，例如调换数据存储中的顺序。

总的来说，不管采用哪种编码方式，对于不同的场景我们也需要找到最优的解码与匹配方案来完成sparse tensor的运算。在之后的篇幅讨论不同PE的workload balance时，还会再考虑到本文阐述的解码模块。

## 参考文献

[1] S. Dave, R. Baghdadi, T. Nowatzki, S. Avancha, A. Shrivastava and B. Li, "Hardware Acceleration of Sparse and Irregular Tensor Computations of ML Models: A Survey and Insights," in *Proceedings of the IEEE*, vol. 109, no. 10, pp. 1706-1752, Oct. 2021.

[2] Kolda, T.G., Bader, B.W., Tensor Decompositions and Applications, *SIAM review*, vol. 51, no. 3, pp. 455{500, (2009).

[3] H. Kwon, P. Chatarasi, M. Pellauer, A. Parashar, V. Sarkar, and T. Krishna, “Understanding reuse, performance, and hardware cost of DNN dataflow: A data-centric approach,” in *Proc. 52nd Annu. IEEE/ACM Int. Symp. Microarchitecture*, Oct. 2019, pp. 754–768.

[4] J.-F. Zhang, C.-E. Lee, C. Liu, Y. S. Shao, S. W. Keckler, and Z. Zhang, “SNAP: A 1.67–21.55 TOPS/W sparse neural acceleration processor for unstructured sparse deep neural network inference in 16 nm CMOS,” in *Proc. Symp. VLSI Circuits*, 2019, pp. C306–C307.