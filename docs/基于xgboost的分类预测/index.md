# 基于XGBoost的分类预测学习笔记


### 学习知识点概要

  * XGBoost的介绍及应用

  * 基于天气数据集的XGBoost分类实战

### 学习内容

#### XGBoost的介绍

XGBoost是2016年由华盛顿大学陈天奇老师带领开发的一个可扩展机器学习系统，是一个可
供用户轻松解决分类、回归或排序问题的软件包。它内部实现了梯度提升树(GBDT)模型，
并对模型中的算法进行了诸多优化，在取得高精度的同时又保持了极快的速度。

更重要的是，XGBoost在系统优化和机器学习原理方面都进行了深入的考虑。毫不夸张的讲，
XGBoost提供的可扩展性，可移植性与准确性推动了机器学习计算限制的上限，该系统在单台
机器上运行速度比当时流行解决方案快十倍以上，甚至在分布式系统中可以处理十亿级的数据。

* 优点：
  * 简单易用。相对其他机器学习库，用户可以轻松使用XGBoost并获得相当不错的效果。
  * 高效可扩展。在处理大规模数据集时速度快效果好，对内存等硬件资源要求不高。
  * 鲁棒性强。相对于深度学习模型不需要精细调参便能取得接近的效果。
  * XGBoost内部实现提升树模型，可以自动处理缺失值。
* 缺点：
  * 相对于深度学习模型无法对时空位置建模，不能很好地捕获图像、语音、文本等高维数据。
  * 在拥有海量训练数据，并能找到合适的深度学习模型时，深度学习的精度可以遥遥领先XGBoost。

XGBoost原理:

XGBoost是基于CART树的集成模型，它的思想是串联多个决策树模型共同进行决策。基模型是CART回归
树，它有两个特点：（1）CART树，是一颗二叉树。（2）回归树，最后拟合结果是连续值。

那么如何串联呢？XGBoost采用迭代预测误差的方法串联。举个通俗的例子，我们现在需要预测一辆车
价值3000元。我们构建决策树1训练后预测为2600元，我们发现有400元的误差，那么决策树2的训练目
标为400元，但决策树2的预测结果为350元，还存在50元的误差就交给第三棵树……以此类推，每一颗树
用来估计之前所有树的误差，最后所有树预测结果的求和就是最终预测结果！

XGBoost模型可以表示为以下形式，我们约定$ f_t(x) $表示前 t 颗树的和，h_t(x)表示第 t 颗决策树，模型定义如下：
$$ f_t(x)=\sum_{t=1}^T h_t(x) $$	

由于模型递归生成，第t步的模型由第t-1步的模型形成，则可以写成：
$$ f_t(x)=f_{t-1}(x)+h_t(x) $$

XGBoost底层实现了GBDT（Gradient Boosting Decision Tree）算法，并对GBDT算法做了一系列优化：

1. 对目标函数进行了泰勒展示的二阶展开，可以更加高效拟合误差。
2. 提出了一种估计分裂点的算法加速CART树的构建过程，同时可以处理稀疏数据。
3. 提出了一种树的并行策略加速迭代。
4. 为模型的分布式算法进行了底层优化

#### XGBoost重要参数

* 1. eta[默认0.3]

通过为每一颗树增加权重，提高模型的鲁棒性。

典型值为0.01-0.2。

* 2. min_child_weight[默认1]

决定最小叶子节点样本权重和。

这个参数可以避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。

但是如果这个值过高，则会导致模型拟合不充分。

* 3. max_depth[默认6]

这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。

典型值：3-10

* 4.  max_leaf_nodes

树上最大的节点或叶子的数量。

可以替代max_depth的作用。

这个参数的定义会导致忽略max_depth参数。

* 5. gamma[默认0]

在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的
最小损失函数下降值。 这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关。

* 6. max_delta_step[默认0]

这参数限制每棵树权重改变的最大步长。如果这个参数的值为0，那就意味着没有约束。如果它被赋予了某个正值，那么它会让这个算法更加保守。

但是当各类别的样本十分不平衡时，它对分类问题是很有帮助的。

* 7. subsample[默认1]

这个参数控制对于每棵树，随机采样的比例。

减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。

典型值：0.5-1

* 8. colsample_bytree[默认1]

用来控制每棵随机采样的列数的占比(每一列是一个特征)。

典型值：0.5-1

* 9. colsample_bylevel[默认1]

用来控制树的每一级的每一次分裂，对列数的采样的占比。

subsample参数和colsample_bytree参数可以起到相同的作用，一般用不到。

* 10. lambda[默认1]

权重的L2正则化项。(和Ridge regression类似)。

这个参数是用来控制XGBoost的正则化部分的。虽然大部分数据科学家很少用到这个参数，但是这个参数在减少过拟合上还是可以挖掘出更多用处的。

* 11. alpha[默认1]

权重的L1正则化项。(和Lasso regression类似)。

可以应用在很高维度的情况下，使得算法的速度更快。

* 12. scale_pos_weight[默认1]

在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。

#### XGBoost的应用

1. 商店销售额预测
2. 高能物理事件分类
3. web文本分类
4. 用户行为预测
5. 运动检测
6. 广告点击率预测
7. 恶意软件分类
8. 灾害风险预测
9. 线课程退学率预测

#### 基于天气数据集的XGBoost分类实战

* Step1: 库函数导入

* Step2: 数据读取/载入

* Step3: 数据信息简单查看

* Step4: 可视化描述

* Step5: 对离散变量进行编码

* Step6: 利用 XGBoost 进行训练与预测

* Step7: 利用 XGBoost 进行特征选择

* Step8: 通过调整参数获得更好的效果

[天气数据集下载](https://tianchi-media.oss-cn-beijing.aliyuncs.com/DSW/7XGBoost/train.csv)

代码分析如下：

```python
import numpy as np
import pandas as pd
# 绘图函数库
import seaborn as sns
# 为了正确评估模型性能，将数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。
from sklearn.model_selection import train_test_split
# 导入XGBoost模型
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
# 从sklearn库中导入网格调参函数
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib
from matplotlib import pyplot as plt


# 我们利用Pandas自带的read_csv函数读取并转化为DataFrame格式
data = pd.read_csv("../Data/weather/train.csv")

# 利用.info()查看数据的整体信息
print(data.info())
# 用-1填补NaN值
data = data.fillna(-1)

# 利用value_counts函数查看训练集标签的数量
pd.Series(data['RainTomorrow']).value_counts()

# 对于特征进行一些统计描述, 均值、标准差、最小值等
print(data.describe())
# 获得每一列的数据是浮点数的列名
numerical_features = [x for x in data.columns if data[x].dtype == np.float]
# 获得每一列的数据不是浮点数且不是RainTomorrow的列名，RainTomorrow是明天是否下雨的布尔值，即标签值
category_features = [x for x in data.columns if data[x].dtype != np.float and x != 'RainTomorrow']

# 选取三个特征与标签组合的散点可视化，
# 三个参数为：数据、画图的类型选择直方图、色彩（以RainTomorrow值的种类为颜色种类进行画图）
sns.pairplot(data=data[['Rainfall', 'Evaporation', 'Sunshine'] + ['RainTomorrow']],
             diag_kind='hist', hue='RainTomorrow')
# 弹出画好的图
plt.show()

# 获取每一列的数据是浮点数的列名
for col in data[numerical_features].columns:
    if col != 'RainTomorrow':
        # 画箱型图，x轴为RainTomorrow值，y轴为col值，saturation为色彩饱和度
        sns.boxplot(x='RainTomorrow', y=col,
                    saturation=0.5, palette='pastel', data=data)
        plt.title(col)  # 设置图上方的标题
        plt.show()

tlog = {}
for i in category_features:
    # 利用value_counts函数查看在训练集标签为yes中，
    # 列表category_features里的每个列名下数据的数量个数
    tlog[i] = data[data['RainTomorrow'] == 'Yes'][i].value_counts()
flog = {}
for i in category_features:
    # 利用value_counts函数查看在训练集标签为no中，
    # 列表category_features里的每个列名下数据的数量个数
    flog[i] = data[data['RainTomorrow'] == 'No'][i].value_counts()

# 准备一张10x10的图，准备作画
plt.figure(figsize=(10, 10))
# 在一行两列的图中，第一个位置（大小为10x5）作画
plt.subplot(1, 2, 1)
# 在第一个位置的图写上标题RainTomorrow
plt.title('RainTomorrow')
# 画上条形图，x值为tlog（明天下雨）中的地区值数量大小进行排序后的值，
# y为对应排序后的地区值，颜色为红色
sns.barplot(x=pd.DataFrame(tlog['Location']).sort_index()['Location'],
            y=pd.DataFrame(tlog['Location']).sort_index().index,
            color="red")
# 在一行两列的图中，第二个位置（大小为10x5）作画
plt.subplot(1, 2, 2)
# 在第一个位置的图写上标题
plt.title('Not RainTomorrow')
# 画上条形图，x值为flog（明天不下雨）中的地区值数量大小进行排序后的值，
# y为对应排序后的地区值，颜色为蓝色
sns.barplot(x=pd.DataFrame(flog['Location']).sort_index()['Location'],
            y=pd.DataFrame(flog['Location']).sort_index().index,
            color="blue")
plt.show()

# 准备一张10x2的图，准备作画
plt.figure(figsize=(10, 2))
# 在一行两列的图中，第一个位置（大小为10x1）作画
plt.subplot(1, 2, 1)
# 在第一个位置的图写上标题RainToday
plt.title('RainToday')
# 画上条形图，x值为tlog（今天下雨）中的地区值数量大小进行排序后的值，
# y为对应排序后的地区值，颜色为红色
sns.barplot(x=pd.DataFrame(tlog['RainToday'][:2]).sort_index()['RainToday'],
            y=pd.DataFrame(tlog['RainToday'][:2]).sort_index().index,
            color="red")
plt.subplot(1, 2, 2)
plt.title('Not RainToday')
# 画上条形图，x值为tlog（今天不下雨）中的地区值数量大小进行排序后的值，
# y为对应排序后的地区值，颜色为蓝色
sns.barplot(x=pd.DataFrame(flog['RainToday'][:2]).sort_index()['RainToday'],
            y=pd.DataFrame(flog['RainToday'][:2]).sort_index().index,
            color="blue")
plt.show()


# 把所有的相同类别的特征编码为同一个值，即将字符串编码成数字
def get_mapfunction(x):
    mapp = dict(zip(x.unique().tolist(), range(len(x.unique().tolist()))))
    def mapfunction(y):
        if y in mapp:
            return mapp[y]
        else:
            return -1
    return mapfunction


# 把不是浮点数且不是RainTomorrow的每列的数据中所有的相同类别的特征编码为同一个值
# 首先执行get_mapfunction()函数，然后返回mapfunction()函数，最后将data[i]中
# 的各个数apply()到mapfunction()函数
for i in category_features:
    data[i] = data[i].apply(get_mapfunction(data[i]))

# 编码后的字符串特征变成了数字
data['Location'].unique()

# 选择其类别为0和1的样本（即是否下雨，No和Yes）
data_target_part = data['RainTomorrow']
data_features_part = data[[x for x in data.columns if x != 'RainTomorrow']]

# 测试集大小为20%，训练集大小为80%
x_train, x_test, y_train, y_test = train_test_split(data_features_part,
                                                    data_target_part,
                                                    test_size=0.2,
                                                    random_state=2020)

# --------------------使用默认参数进行模型训练---------------------------
# 定义 XGBoost模型
clf = XGBClassifier()
# 在训练集上训练XGBoost模型
clf.fit(x_train, y_train)
# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

# 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is:',
      metrics.accuracy_score(y_train, train_predict))  # 0.89824767039793
print('The accuracy of the Logistic Regression is:',
      metrics.accuracy_score(y_test, test_predict))    # 0.85751793333020

# 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
print('The confusion matrix result:\n', confusion_matrix_result)

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
    ax2 = plot_importance(model, importance_type="weight")
    ax2.set_title('weight')
    ax3 = plot_importance(model, importance_type="cover")
    ax3.set_title('cover')
    plt.show()


def classes(data, label, test):
    model = XGBClassifier()
    model.fit(data, label)
    ans = model.predict(test)
    estimate(model, data)
    return ans


ans = classes(x_train, y_train, x_test)
pre = accuracy_score(y_test, ans)
print('acc=', accuracy_score(y_test, ans))

# ------------------以下进行网格搜索并用最优参数进行训练------------------------
# 定义参数取值范围
learning_rate = [0.1, 0.3, 0.6]
subsample = [0.8, 0.9]
colsample_bytree = [0.6, 0.8]
max_depth = [3, 5, 8]

parameters = {'learning_rate': learning_rate,
              'subsample': subsample,
              'colsample_bytree': colsample_bytree,
              'max_depth': max_depth}
model = XGBClassifier(n_estimators=50)

# 进行网格搜索
clf_grid = GridSearchCV(model, parameters, cv=3, scoring='accuracy',
                        verbose=1, n_jobs=-1)
clf_grid = clf_grid.fit(x_train, y_train)
# 网格搜索后的最好参数为
print(clf_grid.best_params_)
# 在训练集和测试集上分布利用最好的模型参数进行预测
# 定义带参数的 XGBoost模型
clf_grid = XGBClassifier(colsample_bytree=0.6, learning_rate=0.1,
                         max_depth=8, subsample=0.8)
# 在训练集上训练XGBoost模型
clf_grid.fit(x_train, y_train)

train_predict = clf_grid.predict(x_train)
test_predict = clf_grid.predict(x_test)

# 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果,
# 应用网格搜索之后准确率稍有提高，但不明显
print('The accuracy of the Logistic Regression is:',
      metrics.accuracy_score(y_train, train_predict))  # 0.89923225693019
print('The accuracy of the Logistic Regression is:',
      metrics.accuracy_score(y_test, test_predict))    # 0.85676778095550

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

1. XGBoost与LightGBM进行比较，有什么不同？
	
   答：获得的精度是差不多的，几乎一样，但是，LightGBM比XGBoost训练得更快且内存消耗更少。

2. XGBoost底层的树结构怎么运作？


### 学习思考与总结

通过此次得学习，我学到了XGBoost的基本原理及其相关应用。但是仍然有其局限性，训练速度较慢，内存消耗较大。

