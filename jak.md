- 模型参数是模型内部的配置变量，可以用数据估计模型参数的值；模型超参数是模型外部的配置，必须手动设置参数的值。
- 此函数中，输入的三个参数分别的含义如下：
    - n_channel=3：信道个数为３
    - n_classes=10：分类的类别共有１０个
    - image_size=24：图片的大小
    - 对self.images、self.labels、self.keep_prob用了占位符函数tf.placeholder()进行初始化赋值。
    - 对self.global_step使用了tf.Variable()函数进行初始化。
'''
def __init__(self, n_channel=3, n_classes=10, image_size=24):
        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[None, image_size, image_size, n_channel], name='images')
        self.labels = tf.placeholder(
            dtype=tf.int64, shape=[None], name='labels')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        self.global_step = tf.Variable( 
            0, dtype=tf.int32, name='global_step')
'''

### 优化器
优化器：具体作用是用来随着epoch的增加，使得学习率不断地进行变化，从而提升模型性能
上述代码主要实现了变化学习率的技术，在前50000个batch使用0.01的学习率，之后50000~100000个batch使用0.001的学习率，之后的学习率降到0.0001
'''
lr = tf.cond(tf.less(self.global_step, 50000),
             lambda: tf.constant(0.01),
             lambda: tf.cond(tf.less(self.global_step, 100000),
                             lambda: tf.constant(0.001),
                             lambda: tf.constant(0.0001)))
self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            self.avg_loss, global_step=self.global_step)
'''

### 数据增强
- 数据增强：主要是在训练数据上增加微小的扰动或者变化，一方面可以增加训练数据，从而提升模型的泛化能力，另一方面可以增加噪声数据，从而增强模型的健壮性。

- 主要的数据增强方法有：翻转变换 flip、随机修剪（random crop）、色彩抖动（color jittering）、平移变换（shift）、尺度变换（scale）、对比度变换（contrast）、噪声扰动（noise）、旋转变换/反射变换 （rotation/reflection）等
    - 图像切割 (_image_crop)：生成比图像尺寸小一些的矩形框，对图像进行随机的切割，最终以矩形框内的图像作为训练数据。
    - 图像翻转 (_image_flip)：随机对图像进行左右翻转。
    - 图像白化：对图像进行白化操作，即将图像本身归一化成 Gaussian(0,1) 分布。
    - 图像噪声：防止过拟合

'''
def data_augmentation(self, images, mode='train', flip=False,
                      crop=False, crop_shape=(24,24,3), whiten=False,
                      noise=False, noise_mean=0, noise_std=0.01):
    # 图像切割
    if crop:
        if mode == 'train':           
            images = self._image_crop(images, shape=crop_shape)
        elif mode == 'test':
            images = self._image_crop_test(images, shape=crop_shape)
    # 图像翻转
    if flip:
        images = self._image_flip(images)
    # 图像白化
    if whiten:
        images = self._image_whitening(images)
    # 图像噪声
    if noise:
        images = self._image_noise(images, mean=noise_mean, std=noise_std)
            
    return images                                                              
def _image_crop(self, images, shape):
    # 图像切割
    new_images = []
    for i in range(images.shape[0]):
        old_image = images[i,:,:,:]
        # 返回左闭右开区间[0,old_image.shape[0] - shape[0] + 1）上离散均匀分布的整数值
        left = numpy.random.randint(old_image.shape[0] - shape[0] + 1)
        top = numpy.random.randint(old_image.shape[1] - shape[1] + 1)
        new_image = old_image[left: left+shape[0], top: top+shape[1], :]
        new_images.append(new_image)
        
    return numpy.array(new_images)
        
    
def _image_crop_test(self, images, shape):
    # 图像切割
    new_images = []
    for i in range(images.shape[0]):        
        old_image = images[i,:,:,:]
        left = int((old_image.shape[0] - shape[0]) / 2)
        top = int((old_image.shape[1] - shape[1]) / 2)
        new_image = old_image[left: left+shape[0], top: top+shape[1], :]
        new_images.append(new_image)
        
    return numpy.array(new_images)
        
    
def _image_flip(self, images):
    # 图像翻转
    for i in range(images.shape[0]):
        old_image = images[i,:,:,:]
        # numpy.random.random()随机生成在[0.0,1.0]中的浮点数
        if numpy.random.random() < 0.5:
            # cv2.flip(old_image, 1)对图像进行水平翻转
            new_image = cv2.flip(old_image, 1)
        else:
            new_image = old_image
        images[i,:,:,:] = new_image
        
    return images
    
def _image_whitening(self, images):
    # 图像白化，即归一化处理
    for i in range(images.shape[0]):
        old_image = images[i,:,:,:]
        # numpy.mean()求均值；　numpy.std()求标准差；
        new_image = (old_image - numpy.mean(old_image)) / numpy.std(old_image)
        images[i,:,:,:] = new_image
        
    return images
    
def _image_noise(self, images, mean=0, std=0.01):
    # 图像噪声
    for i in range(images.shape[0]):
        old_image = images[i,:,:,:]
        new_image = old_image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    # 生成随机数，将其加到像素值上，这里使用的是高斯噪声
                    new_image[i, j, k] += random.gauss(mean, std)
        images[i,:,:,:] = new_image
        
    return images
'''