---
weight: 4
title: "基于LightGBM的分类预测学习笔记"
date: 2021-06-04T17:57:40+08:00
lastmod: 2021-06-04T18:45:40+08:00
draft: false
author: "yingfengwu"
authorLink: "https://yingfengwu.github.io"
description: "这篇文章介绍了LightGBM的分类预测."
resources:
- name: "featured-image"
  src: "featured-image.png"

tags: ["机器学习", "分类", “预测”, "LightGBM"]
categories: ["天池AI训练营"]

lightgallery: true
---

### 学习知识点概要

  * LightGBM的介绍及应用

  * 基于英雄联盟数据集的LightGBM分类实战

### 学习内容

#### LightGBM的介绍

LightGBM是2017年由微软推出的可扩展机器学习系统，是微软旗下DMKT的一个开源项目，由2014年首届
阿里巴巴大数据竞赛获胜者之一柯国霖老师带领开发。它是一款基于GBDT（梯度提升决策树）算法的分
布式梯度提升框架，为了满足缩短模型计算时间的需求，LightGBM的设计思路主要集中在减小数据对内
存与计算性能的使用，以及减少多机器并行计算时的通讯代价。

LightGBM可以看作是XGBoost的升级豪华版，在获得与XGBoost近似精度的同时，又提供了更快的训练速
度与更少的内存消耗。正如其名字中的Light所蕴含的那样，LightGBM在大规模数据集上跑起来更加优
雅轻盈，一经推出便成为各种数据竞赛中刷榜夺冠的神兵利器。

* 优点：
  * 简单易用。提供了主流的Python\C++\R语言接口，用户可以轻松使用LightGBM建模并获得相当不错的效果。
  * 高效可扩展。在处理大规模数据集时高效迅速、高准确度，对内存等硬件资源要求不高。
  * 鲁棒性强。相较于深度学习模型不需要精细调参便能取得近似的效果。
  * LightGBM直接支持缺失值与类别特征，无需对数据额外进行特殊处理
* 缺点：
  * 相对于深度学习模型无法对时空位置建模，不能很好地捕获图像、语音、文本等高维数据。
  * 在拥有海量训练数据，并能找到合适的深度学习模型时，深度学习的精度可以遥遥领先LightGBM。

**LightGBM原理:**

LightGBM是基于CART树的集成模型，它的思想是串联多个决策树模型共同进行决策。基模型是CART回归
树，它有两个特点：（1）CART树，是一颗二叉树。（2）回归树，最后拟合结果是连续值。

那么如何串联呢？LightGBM采用迭代预测误差的方法串联。举个通俗的例子，我们现在需要预测一辆车
价值3000元。我们构建决策树1训练后预测为2600元，我们发现有400元的误差，那么决策树2的训练目
标为400元，但决策树2的预测结果为350元，还存在50元的误差就交给第三棵树……以此类推，每一颗树
用来估计之前所有树的误差，最后所有树预测结果的求和就是最终预测结果！

XGBoost模型可以表示为以下形式，我们约定$ f_t(x) $表示前 t 颗树的和，h_t(x)表示第 t 颗决策树，模型定义如下：
$$ f_t(x)=\sum_{t=1}^T h_t(x) $$	

由于模型递归生成，第t步的模型由第t-1步的模型形成，则可以写成：
$$ f_t(x)=f_{t-1}(x)+h_t(x) $$

LightGBM底层实现了GBDT（Gradient Boosting Decision Tree）算法，并且添加了一系列的新特性：

1. 基于直方图算法进行优化，使数据存储更加方便、运算更快、鲁棒性强、模型更加稳定等。
2. 提出了带深度限制的 Leaf-wise 算法，抛弃了大多数GBDT工具使用的按层生长 (level-wise) 
的决策树生长策略，而使用了带有深度限制的按叶子生长策略，可以降低误差，得到更好的精度。
3. 提出了单边梯度采样算法，排除大部分小梯度的样本，仅用剩下的样本计算信息增益，它是一种在减少数据量和保证精度上平衡的算法。
4. 提出了互斥特征捆绑算法，高维度的数据往往是稀疏的，这种稀疏性启发我们设计一种无损的方法来
减少特征的维度。通常被捆绑的特征都是互斥的（即特征不会同时为非零值，像one-hot），这样两个特征捆绑起来就不会丢失信息。
5. 直接支持类别特征(Categorical Feature)，即不需要进行one-hot编码
6. Cache命中率优化

#### LightGBM重要参数

##### 基本参数调整
1. num_leaves参数 这是控制树模型复杂度的主要参数，一般的我们会使num_leaves小于（2的max_depth次方），
以防止过拟合。由于LightGBM是leaf-wise建树与XGBoost的depth-wise建树方法不同，num_leaves比depth有更大的作用。
2. min_data_in_leaf 这是处理过拟合问题中一个非常重要的参数. 它的值取决于训练数据的样本个树和 num_leaves参
数. 将其设置的较大可以避免生成一个过深的树, 但有可能导致欠拟合. 实际应用中, 对于大数据集, 设置其为几百或几千就足够了。
3. max_depth 树的深度，depth 的概念在 leaf-wise 树中并没有多大作用, 因为并不存在一个从 leaves 到 depth 的合理映射。

##### 针对训练速度的参数调整

1. 通过设置 bagging_fraction 和 bagging_freq 参数来使用 bagging 方法。
2. 通过设置 feature_fraction 参数来使用特征的子抽样。
3. 选择较小的 max_bin 参数。
4. 使用 save_binary 在未来的学习过程对数据加载进行加速。

##### 针对准确率的参数调整
1. 使用较大的 max_bin （学习速度可能变慢）
1. 使用较小的 learning_rate 和较大的 num_iterations
1. 使用较大的 num_leaves （可能导致过拟合）
1. 使用更大的训练数据
1. 尝试 dart 模式

##### 针对过拟合的参数调整
1. 使用较小的 max_bin
1. 使用较小的 num_leaves
1. 使用 min_data_in_leaf 和 min_sum_hessian_in_leaf
1. 通过设置 bagging_fraction 和 bagging_freq 来使用 bagging
1. 通过设置 feature_fraction 来使用特征子抽样
1. 使用更大的训练数据
1. 使用 lambda_l1, lambda_l2 和 min_gain_to_split 来使用正则
1. 尝试 max_depth 来避免生成过深的树

#### LightGBM的应用

1. 金融风控
2. 购买行为识别
3. 交通流量预测
4. 环境声音分类
5. 基因分类
6. 生物成分分析

#### 基于英雄联盟数据集的LightGBM分类实战

* Step1: 库函数导入

* Step2: 数据读取/载入

* Step3: 数据信息简单查看

* Step4: 可视化描述

* Step5: 利用 XGBoost 进行训练与预测

* Step6: 利用 XGBoost 进行特征选择

* Step7: 通过调整参数获得更好的效果

[英雄联盟数据集下载](https://tianchi-media.oss-cn-beijing.aliyuncs.com/DSW/8LightGBM/high_diamond_ranked_10min.csv)

代码分析如下：

```python
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import accuracy_score
from lightgbm import plot_importance
from sklearn.model_selection import GridSearchCV
# 绘图函数库
import matplotlib.pyplot as plt
import seaborn as sns

# 我们利用Pandas自带的read_csv函数读取并转化为DataFrame格式
df = pd.read_csv('../Data/LOL/high_diamond_ranked_10min.csv')
# 利用.info()查看数据的整体信息
print(df.info())
# 进行简单的数据查看，我们可以利用.head()头部.tail()尾部
print(df.head())
print(df.tail())

# 标注标签并利用value_counts函数查看训练集标签的数量
y = df.blueWins
print(y.value_counts())
# 标注特征列，drop_cols中存放非特征列，然后丢弃
drop_cols = ['gameId', 'blueWins']
x = df.drop(drop_cols, axis=1)
# 对于特征进行一些统计描述
print(x.describe())

# 根据上面的描述，我们可以去除一些重复变量，比如只要知道蓝队是否拿到一血，
# 我们就知道红队有没有拿到，可以去除红队的相关冗余数据。
drop_cols = ['redFirstBlood','redKills','redDeaths',
             'redGoldDiff','redExperienceDiff', 'blueCSPerMin',
             'blueGoldPerMin','redCSPerMin','redGoldPerMin']
x.drop(drop_cols, axis=1, inplace=True)

# 减去平均数除以标准差相当于对原始数据进行了线性变换，没有改变数据之间的相
# 对位置，也没有改变数据的分布，只是数据的平均数变成0，标准差变成1。根本不
# 会变成正态分布，除非它本来就是。
data = x
data_std = (data - data.mean()) / data.std()
data = pd.concat([y, data_std.iloc[:, 0:9]], axis=1)
data = pd.melt(data, id_vars='blueWins',
               var_name='Features', value_name='Values')

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# 绘制小提琴图
sns.violinplot(x='Features', y='Values', hue='blueWins', data=data, split=True,
               inner='quart', ax=ax[0], palette='Blues')
fig.autofmt_xdate(rotation=45)  # 将x轴每列的标号旋转45°

data = x
data_std = (data - data.mean()) / data.std()
data = pd.concat([y, data_std.iloc[:, 9:18]], axis=1)
data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')

# 绘制小提琴图
sns.violinplot(x='Features', y='Values', hue='blueWins',
               data=data, split=True, inner='quart', ax=ax[1], palette='Blues')
fig.autofmt_xdate(rotation=45)
plt.show()

plt.figure(figsize=(18,14))
sns.heatmap(round(x.corr(),2), cmap='Blues', annot=True)
plt.show()

# 去除冗余特征
drop_cols = ['redAvgLevel','blueAvgLevel']
x.drop(drop_cols, axis=1, inplace=True)

sns.set(style='whitegrid', palette='muted')

# 构造两个新特征
x['wardsPlacedDiff'] = x['blueWardsPlaced'] - x['redWardsPlaced']
x['wardsDestroyedDiff'] = x['blueWardsDestroyed'] - x['redWardsDestroyed']

data = x[['blueWardsPlaced','blueWardsDestroyed',
          'wardsPlacedDiff','wardsDestroyedDiff']].sample(1000)
data_std = (data - data.mean()) / data.std()
data = pd.concat([y, data_std], axis=1)
data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')

plt.figure(figsize=(10,6))
sns.swarmplot(x='Features', y='Values', hue='blueWins', data=data)
plt.xticks(rotation=45)
plt.show()

# 去除和眼位相关的特征
drop_cols = ['blueWardsPlaced','blueWardsDestroyed','wardsPlacedDiff',
             'wardsDestroyedDiff','redWardsPlaced','redWardsDestroyed']
x.drop(drop_cols, axis=1, inplace=True)

x['killsDiff'] = x['blueKills'] - x['blueDeaths']
x['assistsDiff'] = x['blueAssists'] - x['redAssists']

x[['blueKills','blueDeaths','blueAssists',
   'killsDiff','assistsDiff','redAssists']].hist(figsize=(12,10), bins=20)
plt.show()

data = x[['blueKills','blueDeaths','blueAssists',
          'killsDiff','assistsDiff','redAssists']].sample(1000)
data_std = (data - data.mean()) / data.std()
data = pd.concat([y, data_std], axis=1)
data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')

plt.figure(figsize=(10,6))
sns.swarmplot(x='Features', y='Values', hue='blueWins', data=data)
plt.xticks(rotation=45)
plt.show()

data = pd.concat([y, x], axis=1).sample(500)

sns.pairplot(data, vars=['blueKills','blueDeaths','blueAssists',
                         'killsDiff','assistsDiff','redAssists'],
             hue='blueWins')
plt.show()

x['dragonsDiff'] = x['blueDragons'] - x['redDragons']
x['heraldsDiff'] = x['blueHeralds'] - x['redHeralds']
x['eliteDiff'] = x['blueEliteMonsters'] - x['redEliteMonsters']

data = pd.concat([y, x], axis=1)

eliteGroup = data.groupby(['eliteDiff'])['blueWins'].mean()
dragonGroup = data.groupby(['dragonsDiff'])['blueWins'].mean()
heraldGroup = data.groupby(['heraldsDiff'])['blueWins'].mean()

fig, ax = plt.subplots(1,3, figsize=(15,4))

eliteGroup.plot(kind='bar', ax=ax[0])
dragonGroup.plot(kind='bar', ax=ax[1])
heraldGroup.plot(kind='bar', ax=ax[2])

print(eliteGroup)
print(dragonGroup)
print(heraldGroup)

plt.show()

x['towerDiff'] = x['blueTowersDestroyed'] - x['redTowersDestroyed']

data = pd.concat([y, x], axis=1)

towerGroup = data.groupby(['towerDiff'])['blueWins']
print(towerGroup.count())
print(towerGroup.mean())

figure, ax = plt.subplots(1, 2, figsize=(15, 5))

towerGroup.mean().plot(kind='line', ax=ax[0])
ax[0].set_title('Proportion of Blue Wins')
ax[0].set_ylabel('Proportion')

towerGroup.count().plot(kind='line', ax=ax[1])
ax[1].set_title('Count of Towers Destroyed')
ax[1].set_ylabel('Count')

# ---------------------------------利用 LightGBM 进行训练与预测----------------------------------
# 为了正确评估模型性能，将数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。
# 选择其类别为0和1的样本
data_target_part = y
data_features_part = x

# 测试集大小为20%，训练集大小为80%
x_train, x_test, y_train, y_test = train_test_split(data_features_part,
                                                    data_target_part,
                                                    test_size = 0.2,
                                                    random_state = 2020)

# 定义 LightGBM 模型
clf = LGBMClassifier()
# 在训练集上训练LightGBM模型
clf.fit(x_train, y_train)

# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

# 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is:',
      metrics.accuracy_score(y_train,train_predict))
print('The accuracy of the Logistic Regression is:',
      metrics.accuracy_score(y_test,test_predict))

# 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
print('The confusion matrix result:\n',confusion_matrix_result)

# 利用热力图对于结果进行可视化
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

sns.barplot(y=data_features_part.columns, x=clf.feature_importances_)


# -------------------绘制对应列（即特征）的重要性图-----------------------
def estimate(model, data):
    # sns.barplot(data.columns,model.feature_importances_)
    ax1 = plot_importance(model, importance_type="gain")
    ax1.set_title('gain')
    ax2 = plot_importance(model, importance_type="split")
    ax2.set_title('split')
    plt.show()


def classes(data, label, test):
    model = LGBMClassifier()
    model.fit(data, label)
    ans = model.predict(test)
    estimate(model, data)
    return ans


ans = classes(x_train, y_train, x_test)
pre = accuracy_score(y_test, ans)
print('acc=', accuracy_score(y_test, ans))

# 定义参数取值范围
learning_rate = [0.1, 0.3, 0.6]
feature_fraction = [0.5, 0.8, 1]
num_leaves = [16, 32, 64]
max_depth = [-1,3,5,8]

parameters = {'learning_rate': learning_rate,
              'feature_fraction': feature_fraction,
              'num_leaves': num_leaves,
              'max_depth': max_depth}
model = LGBMClassifier(n_estimators = 50)

# 进行网格搜索
clf = GridSearchCV(model, parameters, cv=3,
                   scoring='accuracy', verbose=3, n_jobs=-1)
clf = clf.fit(x_train, y_train)
# 网格搜索后的最好参数为
print(clf.best_params_)
# 在训练集和测试集上分布利用最好的模型参数进行预测
# 定义带参数的 LightGBM模型
clf = LGBMClassifier(feature_fraction=1,
                     learning_rate=0.1,
                     max_depth=3,
                     num_leaves=16)
# 在训练集上训练LightGBM模型
clf.fit(x_train, y_train)

train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

# 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is:',
      metrics.accuracy_score(y_train,train_predict))
print('The accuracy of the Logistic Regression is:',
      metrics.accuracy_score(y_test,test_predict))

# 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
print('The confusion matrix result:\n', confusion_matrix_result)

# 利用热力图对于结果进行可视化
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

```

### 学习问题与解答

1. LightGBM是怎么做到训练速度更快且内存消耗更少得呢？

	对于每个特征，需要扫描所有得数据实例去估计所有可能划分的点的信息增益是很耗时的，为了解决该问题，
	[微软](https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) 相关
	人员提出了两个新的技术：基于梯度的单边采样和独有的特征绑定。用基于梯度的单边采样的方法，可以用
	小梯度排除大多重要的数据实例，并且用剩下的实例去估计信息增益。该文证实了基于梯度的单边采样的方法
	用小规模的数据即可获得相当准确的信息增益估计。该文用独有的特征绑定去相互地绑定独有特征来减少特征
	数量，而且找到最优绑定是NP难问题，但是贪婪算法可以实现很好的近似率。
	
	排他特征绑定:
	
	在稀疏特征空间，许多特征是相互排斥地，也就是它们不会同时为非零值（有些特征必为零，有些不为零）。
	直方图构建的复杂度从O(data,feature)到O(data,bundle)，且bundle值远小于feature值。这里有两个问题
	需要解决，一个是哪些特征需要被绑定在一起，另一个是怎样去构建绑定。
	
	定理1：划分特征到小批量的排他特征的问题是NP难的。
	
	证明：该文将图着色问题归约到该问题中。因为图着色问题是NP难的，所以可以由此推出我们的结论。
	
	




### 学习思考与总结

通过此次得学习，我学到了LightGBM的基本原理及其相关应用，LightGBM训练速度更快且内存消耗更少。
