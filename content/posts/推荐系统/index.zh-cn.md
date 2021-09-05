---
weight: 4
title: "推荐系统介绍"
date: 2021-05-31T17:57:40+08:00
lastmod: 2021-05-31T18:45:40+08:00
draft: false
author: "yingfengwu"
authorLink: "https://yingfengwu.github.io"
description: "这篇文章介绍了Spark相关知识."
resources:
- name: "featured-image"
  src: "featured-image.png"

tags: ["数据分析", "推荐系统"]
categories: ["大数据"]

lightgallery: true
---

Spark是专为大规模数据处理而设计的快速通用的计算引擎。
诞生于2009年，是加州大学伯克利分校RAD实验室的一个研究项目，最初基于Hadoop Mapreduce。
但是，由于Mapreduce在迭代式计算和交互式上低效，因此引入内存存储。

Spark包括多个紧密集成的组件：

- name: "featured-image"
  src: "featured-image.png"

### 推荐系统应用

#### 个性化音乐

1. 基本功能：任务调度，内存管理，容错机制
2. 内部定义了RDDs（Resilient distributed datasets, 弹性分布式数据集）
3. 提供API创建和操作RDDs
4. 在应用场景中，为其他组件提供底层的服务
   
#### 电子商务

1. Spark处理结构化数据的库，类似Hive SQL，Mysql
2. 应用场景，企业用来做报表统计

#### 电影视频

1. 实时数据流处理组件，类似storm
2. 提供API操作实时流数据
3. 应用场景，企业用来从Kafka（等消息队列中）接收数据做实时统计

#### 社交网络

1. 一个包含通用机器学习功能的包，Machine learning lib，包括分类、聚类、回归、模型评估和数据导入等
2. Mlib提供的方法都支持集群上的横向扩展（平时使用python是单机处理且有限的，而Mlib是集群的）
3. 应用场景，机器学习

#### 个性化阅读

1. 处理图的库（如社交网络图），并进行图的并行计算
2. 像Spark Steaming和Spark SQL一样，它也继承了RDDs API
3. 提供了各种图操作和常用的图算法，例如PangeRank算法
4. 应用场景，图计算

#### 位置服务

1. 集群管理，Spark自带的一个集群管理是单独调度器
2. 常见集群管理包括Hadoop YARN，Apache Mesos

#### 个性化邮件

1. Spark底层优化了，基于Spark底层的组件也得到了相应的优化
2. 紧密集成，节省了各个组件组合使用时的部署，测试等时间
3. 向Spark增加新的组件时，其它组件可立刻享用新组件的功能

### 个性化广告

1. Spark应用场景：时效性要求高（因为基于内存）、机器学习领域
2. Spark不具有HDFS（分布式文件系统）的存储能力，要借助HDFS等工具来持久化数据
3. Hadoop应用场景：离线处理、对时效性要求不高

### 个性化旅游

1. Spark应用场景：时效性要求高（因为基于内存）、机器学习领域
2. Spark不具有HDFS（分布式文件系统）的存储能力，要借助HDFS等工具来持久化数据
3. Hadoop应用场景：离线处理、对时效性要求不高

### 证券、投资

1. Spark应用场景：时效性要求高（因为基于内存）、机器学习领域
2. Spark不具有HDFS（分布式文件系统）的存储能力，要借助HDFS等工具来持久化数据
3. Hadoop应用场景：离线处理、对时效性要求不高


### 推荐系统分类

#### 根据实时性分类

**1. 离线推荐**

1.1 下载地址：http://spark.apache.org/downloads.html，
   
   版本需匹配：Spark 1.6.2 - Scala 2.10 或 Spark 2.0.0 - Scala 2.11
		
1.2 解压

**2. 实时推荐**
	
2.1 目录：
	

#### 根据推荐是否个性化分类

**1. 基于统计的推荐**

1.1 下载地址：http://spark.apache.org/downloads.html，
   
   版本需匹配：Spark 1.6.2 - Scala 2.10 或 Spark 2.0.0 - Scala 2.11
		
1.2 解压

**2. 个性化推荐**
	
2.1 目录：

#### 根据推荐原则分类

**1. 基于相似度的推荐**

1.1 下载地址：http://spark.apache.org/downloads.html，
   
   版本需匹配：Spark 1.6.2 - Scala 2.10 或 Spark 2.0.0 - Scala 2.11
		
1.2 解压

**2. 基于知识的推荐**
	
2.1 目录：

**3. 基于模型的推荐**
	
3.1 目录：

#### 根据数据源分类

**1. 基于人口统计学的推荐**

1.1 下载地址：http://spark.apache.org/downloads.html，
   
   版本需匹配：Spark 1.6.2 - Scala 2.10 或 Spark 2.0.0 - Scala 2.11
		
1.2 解压

**2. 基于内容的推荐**
	
2.1 目录：

**3. 基于协同过滤的推荐**
	
3.1 目录：


