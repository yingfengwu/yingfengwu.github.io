---
weight: 4
title: "动态规划总结（多示例+讲解）"
date: 2021-06-01T17:57:40+08:00
lastmod: 2021-06-19T18:45:40+08:00
draft: false
author: "yingfengwu"
authorLink: "https://yingfengwu.github.io"
description: "这篇文章展示了动态规划的总结."
resources:
- name: "featured-image"
  src: "featured-image.png"

tags: ["动态规划"]
categories: ["LeetCode"]

lightgallery: true
---



动态规划功能强大，它能够解决子问题并使用这些答案来解决大问题。但
仅当每个子问题都是离散的，即不依赖于其他子问题时，动态规划才管用。
比如，想去以下地方旅游4天，假设将埃菲尔铁塔加入“背包”后，卢浮宫将
更“便宜”：只要1天时间，而不是1.5天。用动态规划对这种情况建模呢？
这是没办法建模的，因为存在依赖关系。

| 景点 | 停留天数 |  评分 |
| ------ | ----------- | --- |
| 埃菲尔铁塔   | 1.5天 | 8 |
| 卢浮宫 | 1.5天 |  9  |
| 巴黎圣母院    | 1.5天 |  7  |


### 一、斐波那契数列求解

题目:

写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：

F(0) = 0,   F(1) = 1

F(N) = F(N - 1) + F(N - 2), 其中 N > 1.

斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

:joy:

#### 递归法

* 原理： 把f(n)问题的计算拆分成 f(n-1)和f(n−2)两个子问题的计算，并递归，以f(0)和f(1)为终止条件。

* 缺点： 大量重复的递归计算，例如f(n)和f(n - 1)两者向下递归需要 各自计算f(n−2)的值。

```python
class Solution(object): # 该算法n越大执行越慢
    def fib(self, n):
        if n == 0:
			return 0
		elif n == 1:
			return 1
		else
			return self.fib(n-1) + self.fib(n-2)
```

#### 记忆化递归法

* 原理： 在递归法的基础上，新建一个长度为 n 的数组，用于在递归时存储 f(0) 至 f(n) 的
数字值，重复遇到某数字则直接从数组取用，避免了重复的递归计算。

* 缺点： 记忆化存储需要使用 O(N) 的额外空间。

```python
class Solution(object):
    def fib(self, n):
        f = [0]*100  # 定义一定长度的数组
        if n == 1 or n == 0:
            return n
        elif f[n] != 0:
            return f[n]
        else:
            f[n] = self.fib(n-1) + self.fib(n-2)  # 保存至数组f避免重复计算
            return f[n]
```

#### 动态规划法

* 原理： 以斐波那契数列性质 f(n + 1) = f(n) + f(n - 1)为转移方程。从计算效率、空间复杂度上看，动态规划是本题的最佳解法。
  * 状态定义： 设 dp 为一维数组，其中 dp[i] 的值代表 斐波那契数列第 i 个数字 。
  * 转移方程： dp[i + 1] = dp[i] + dp[i - 1] ，即对应数列定义 f(n + 1) = f(n) + f(n - 1)；
  * 初始状态： dp[0] = 0, dp[1]=1 ，即初始化前两个数字；
  * 返回值： dp[n]，即斐波那契数列的第 n 个数字。

```python
class Solution(object):
    def fib(self, n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a
```


### 二、把数字翻译成字符串

题目：

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 
翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有
多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

示例1:

输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"

#### 动态规划法

  * 状态定义： 状态定义： 设动态规划列表 dp，dp[i] 代表以x_i为结尾的数字的翻译方案数量
  * 转移方程： 若 $x_i$和$x_{i-1}$组成的两位数字可被整体翻译，则 dp[i] = dp[i - 1] + dp[i - 2]，否则 dp[i] = dp[i - 1]。
  $$
	dp[i] = 
	\begin{cases} 
	dp[i-1]+dp[i-2], & (10x_{i-1} + x_{i-1})\in[10，25] \\\\ 
	dp[i-1], & (10x_{i-1} + x_i)\in[0,10)\bigcup(25,99]
	\end{cases}
  $$
  * 初始状态： dp[0] = dp[1] = 1，即 “无数字” 和 “第1位数字” 的翻译方法数量均为 1；
  * 返回值： dp[n]，即此数字的翻译方案数量；

第一种写法：

```python
class Solution(object):
    def translateNum(self, num):
        s = str(num)
        # 当 num 第 1, 2位的组成的数字 ∈ [10,25]时，
		# 显然应有 2 种翻译方法，即 dp[2] = dp[1] + dp[0] = 2，
		# 而显然 dp[1] = 1，因此推出 dp[0] = 1，即初始化a=b=1
        a = b = 1  
		for i in range(2, len(s) + 1):
            tmp = s[i - 2:i]
            if "10" <= tmp <= "25":
                c = a + b 
            else:
                c = a
            b = a
            a = c
        return a
```

第二种写法：

```python
class Solution(object):
    def translateNum(self, num):
        s = str(num)
        a = b = 1
        for i in range(2, len(s) + 1):
            a, b = (a + b if "10" <= s[i - 2:i] <= "25" else a), a
        return a
```


### 三、青蛙跳台阶问题求解

题目：

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

#### 动态规划法

  * 状态定义： 设 dp 为一维数组，其中 dp[i] 的值代表斐波那契数列的第 i 个数字。
  * 转移方程： dp[i + 1] = dp[i] + dp[i - 1]，即对应数列定义 f(n + 1) = f(n) + f(n - 1)；
  * 初始状态： dp[0] = 1, dp[1]=1 ，即初始化前两个数字；
  * 返回值： dp[n]，即斐波那契数列的第 n 个数字。


```python
class Solution(object):
    def numWays(self, n):
        if n < 2:
            return 1
        d = [1]*(n+1)
        for i in range(2, n+1):
            d[i] = d[i-1] + d[i-2]
        return d[-1] % 1000000007
```


### 四、连续子数组的最大和

题目：

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

示例1:

输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

#### 动态规划法

* 原理： 以斐波那契数列性质 f(n + 1) = f(n) + f(n - 1)为转移方程。从计算效率、空间复杂度上看，动态规划是本题的最佳解法。
  * 状态定义： 设 dp 为一维数组，其中 dp[i] 的值代表 斐波那契数列第 i 个数字 。
  * 转移方程： dp[i + 1] = dp[i] + dp[i - 1]，即对应数列定义f(n+1)=f(n)+f(n−1) ；
  * 初始状态： dp[0] = 0, dp[1]=1，即初始化前两个数字；
  * 返回值： dp[n] ，即斐波那契数列的第 n 个数字。

```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i in range(1, len(nums)):
            nums[i] += max(nums[i-1], 0)
        return max(nums)
```

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int max_num = nums[0];
        for(int i=1;i<nums.size();i++){
            nums[i] += max(nums[i-1], 0);
            if(nums[i] > max_num){
                max_num = nums[i];
            }
        }
        return max_num;
    }
};
```

