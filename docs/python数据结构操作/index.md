# python数据结构栈和队列的相关方法


s = stack()

stack()函数方法如下：

|      栈操作    |       栈内容       | 返回值 |
| -------------- | ------------------ |  ----  |
|  s.is_empty()   | []                 |  True  |
|  s.push(4)     | [4]                |        |
|  s.push(‘dog’) | [4,’dog’]          |        |
|  s.peek()      | [4,’dog’]          |  ‘dog’ |
|  s.push(True)  | [4,’dog’,True]          |
|  s.size()   | [4,’dog’,True]          |    3   |
|  s.is_empty()  | [4,’dog’,True]          |  False |
|  s.push(8.4)   | [4,’dog’,True,8.4] |
|  s.pop()       | [4,’dog’,True]          |   8.4  |
|  s.pop()       | [4,’dog’]         |  True  |
|  s.size()      | [4,’dog’]          |    2   |

q = Queue()

Queue()函数方法如下：

|      队列操作    |       队列内容       | 返回值 |
| -------------- | ------------------ |  ----  |
|  q=Queue()    | []                 |  Queue 对象  |
|  q.isEmpty()     | []                |   True     |
|  q.enqueue(4) | [4]          |        |
|  q.enqueue('dog’)    | [’dog’, 4]          |   |
|  q.enqueue(True) | [True,‘dog’,4]          |
|  q.size()  | [True,‘dog’,4]          |    3   |
|  q.isEmpty()  | [True,‘dog’,4]          |  False |
|  q.enqueue(8.4)  | [8.4, True,‘dog’,4] |      |
|  q.dequeue()       | [8.4,True,‘dog’]          |   4  |
|  q.dequeue()     | [8.4,True]         |  'dog'  |
|  q.size()      | [8.4,True]          |    2   |

d = Deque()

Deque()函数方法如下：

|      双端队列操作    |      双端队列内容       | 返回值 |
| -------------- | ------------------ |  ----  |
|  d=Deque()   | []                 |  Deque 对象  |
|  d.isEmpty()    | []                |    True    |
|  d.addRear(4) | [4]          |        |
|  d.addRear(‘dog’)   | [’dog’, 4]          |   |
|  d.addFront(‘cat’)  | [’dog’, 4, 'cat']          |
|  d.addFront(True)   | [‘dog’,4, ‘cat’, True]         |       |
|  d.size()   | [‘dog’, 4, ‘cat’, True]          |  4 |
|  d.isEmpty()   | [‘dog’, 4, ‘cat’, True] |   False   |
|  d.addRear(8.4)      | [8.4, 'dog’, 4, ‘cat’,True]          |     |
|  d.removeRear()       | [‘dog’, 4, ‘cat’, True]         |  8.4  |
|  d.removeFront()      | [‘dog’, 4, ‘cat’]          |   True   |

:smile:


