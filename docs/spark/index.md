# Spark


Spark是专为大规模数据处理而设计的快速通用的计算引擎。
诞生于2009年，是加州大学伯克利分校RAD实验室的一个研究项目，最初基于Hadoop Mapreduce。
但是，由于Mapreduce在迭代式计算和交互式上低效，因此引入内存存储。

Spark包括多个紧密集成的组件：

- name: "featured-image"
  src: "featured-image.png"

### Spark的组件

#### Spark core

1. 基本功能：任务调度，内存管理，容错机制
2. 内部定义了RDDs（Resilient distributed datasets, 弹性分布式数据集）
3. 提供API创建和操作RDDs
4. 在应用场景中，为其他组件提供底层的服务
   
#### Spark SQL

1. Spark处理结构化数据的库，类似Hive SQL，Mysql
2. 应用场景，企业用来做报表统计

#### Spark Streaming

1. 实时数据流处理组件，类似storm
2. 提供API操作实时流数据
3. 应用场景，企业用来从Kafka（等消息队列中）接收数据做实时统计

#### Mlib

1. 一个包含通用机器学习功能的包，Machine learning lib，包括分类、聚类、回归、模型评估和数据导入等
2. Mlib提供的方法都支持集群上的横向扩展（平时使用python是单机处理且有限的，而Mlib是集群的）
3. 应用场景，机器学习

#### Graphx

1. 处理图的库（如社交网络图），并进行图的并行计算
2. 像Spark Steaming和Spark SQL一样，它也继承了RDDs API
3. 提供了各种图操作和常用的图算法，例如PangeRank算法
4. 应用场景，图计算

#### Cluster Managers

1. 集群管理，Spark自带的一个集群管理是单独调度器
2. 常见集群管理包括Hadoop YARN，Apache Mesos

#### 紧密集成的优点

1. Spark底层优化了，基于Spark底层的组件也得到了相应的优化
2. 紧密集成，节省了各个组件组合使用时的部署，测试等时间
3. 向Spark增加新的组件时，其它组件可立刻享用新组件的功能

### Spark与Hadoop比较

1. Spark应用场景：时效性要求高（因为基于内存）、机器学习领域
2. Spark不具有HDFS（分布式文件系统）的存储能力，要借助HDFS等工具来持久化数据
3. Hadoop应用场景：离线处理、对时效性要求不高

### 安装

#### Spark安装

**1. Spark下载，安装**

1.1 下载地址：http://spark.apache.org/downloads.html，
   
   版本需匹配：Spark 1.6.2 - Scala 2.10 或 Spark 2.0.0 - Scala 2.11
		
1.2 解压

**2. Spark Shell操作**
	
2.1 目录：
	
bin包含用来和Spark交互的可执行文件，如Spark shell
		
core，steaming，python，...包含主要组件的源代码

examples包含一些单机Spark job，可以用来研究和运行的例子
	
2.2 Spark的Shell
	
能够处理分布在集群上的数据
		
Spark把数据加载到节点的内存中，因此分布式处理可在秒级完成
		
快速迭代式计算，实时查询、分析一般能够在shells中完成
	  
Spark提供了Python shells和Scala shells

**3. Spark开发环境搭建**
	
3.1 IntelliJ IDEA的下载、安装：
	
下载地址：https://www.jetbrains.com/idea/
	
3.2 插件安装
	
在IntelliJ IDEA开发环境中安装即可
	
3.3 搭建开发环境常遇到的问题
	
网络问题
	
版本匹配问题：Scala2.10.5，Jdk1.8，Spark1.6.2，Sbt0.13.8 

#### Scala安装

1. Spark下载，安装

   下载地址：http://www.scala-lang.org/download/2.10.5.html
   
   版本需匹配：Spark 1.6.2 - Scala 2.10 或 Spark 2.0.0 - Scala 2.11
	
2. Scala基础知识
   
   在Scala创建变量时，必须使用val或var
   
   val，变量值不可修改，一旦分配不能重新指向别的值
   
   var，分配后，可以指向类型相同的值

### RDDs介绍

通过SparkContext对象访问Spark, SparkContext对象代表和一个集群的连接, 在Shell中SparkContext自动创建好了，就是sc

1. RDDs并行的分布在整个集群中，如将500G的一个执行文件划分成5个100G的文件到不同的机器并行

2. RDDs是Spark分发数据和计算的基础抽象类

3. 一个RDD是一个不可改变的分布式集合对象

4. Spark中，所有的计算都是通过RDDs的创建，转换，操作完成的

5. 一个RDD内部由许多partitions（分片）组成，

   分片：

   每个分片包括一部分数据，partitions可在集群不同节点上计算

   分片是Spark并行处理的单元，Spark顺序的，并行的处理分片

6. RDDs创建方法：
   
   把一个存在的集合传给SparkContext的parallelize方法，测试用
   
   val rdd = sc.parallelize(Array(1,2,2,4),4)
   
   第一个参数：待并行化处理的集合，第二个参数：分区个数
   
   加载外部数据集：
   
   val = rddText = sc。textFile("helloSpark.txt")



