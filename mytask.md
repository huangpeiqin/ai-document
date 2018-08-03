### 权重衰减

​	对于目标函数加入正则化项，限制权重参数的个数，这是一种防止过拟合的方法。用训练集对模型进行训练时，如果不对参数加以限制，虽然对于训练集来说，模型的识别准确率会越来越高，但换成其他数据集就识别率就会下降，这种情况就叫过拟合。或者说是为了提高泛化能力。

​	调用实例：

```python
if weight_decay:
	weight_decay = tf.multiply(tf.nn.l2_loss(self.weight), self.weight_decay)
    tf.add_to_collection('losses', weight_decay)
```

​	上述事例中，先将weight（权重）传入损失函数中，然后与初始的weight decay做乘法作为新的权重系数，接着将其与'losses'字符串一起放入集合，变成list。正则项一般指模型的复杂度，weight decay可以调节模型复杂度对损失函数的影响，weight decay越大，复杂模型损失函数的值就会越大。

### dropout

​	在每次训练的时候，让某些的特征检测器停过工作，即让神经元以一定的概率不被激活，这样可以防止过拟合，提高泛化能力。目的和权重衰减一模一样，只是方法不同，权重衰减是通过限制参数，而dropout是限制每次激活的神经元数量。

​	调用实例：

```python
if self.dropout:
	self.hidden = tf.nn.dropout(self.hidden, keep_prob=self.keep_prob)	
```

​	self.hidden为训练或者测试数据，keep_porb是倍率，停止的神经元数据会变为0，而其他的则变成原来的1/keep_prob倍。每次停止的神经元一般通过随机算法选出来，从动机论的角度来讲，随机搭配工作的神经元，能够减少相互之间的依赖性，泛化能力相应就会提高了。

### 批正则化

​	batch normalization对神经网络的每一层的输入数据都进行正则化处理，这样有利于让数据的分布更加均匀，不会出现所有数据都会导致神经元的激活，或者所有数据都不会导致神经元的激活，这是一种数据标准化方法，能够提升模型的拟合能力。

```python
if self.batch_normal:
	mean, variance = tf.nn.moments(intermediate, axes=[0])
    self.hidden = tf.nn.batch_normalization(
    	intermediate, mean, variance, self.bias, self.gamma, self.epsilon)
else:
   	self.hidden = intermediate + self.bias
```

​	首先计算统计矩，mean为一阶矩，也就是均值，variance为二阶中心矩即方差，axes[0]表示按列计算，随后将这两个数据作为参数调用batch normalization函数，返回为输入。批正则化不仅能提升模型的拟合能力，而且加大探索步长，加快收敛速度。