# Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Monodepth data loader.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf

def string_length_tf(t):
    """获取图片路径的长度,自定义 python 函数"""
    # tf.py_func: 包装一个普通的 Python 函数，这个函数接受 numpy 的数组作为输入和输出，
    # 让这个函数可以作为 TensorFlow 计算图上的计算节点 OP 来使用
    return tf.py_func(len, [t], [tf.int64])

class MonodepthDataloader(object):
    """monodepth dataloader"""

    def __init__(self, data_path, filenames_file, params, dataset, mode):
        self.data_path = data_path
        self.params = params
        self.dataset = dataset
        self.mode = mode

        # 自定义self.left_image_batch， self.right_image_batch的值
        self.left_image_batch  = None
        self.right_image_batch = None

        # 直接从文件中读取数据
        # tf.train.string_input_producer函数把我们需要的全部文件打包为一个tf内部的queue类型
        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        # 这个tensor做些数据与处理,分开image和label数据
        _, line = line_reader.read(input_queue)

        # 将[line]按照delimiter 拆分为多个元素，返回值为一个SparseTensor
        # 假如有两个字符串，source[0]是“hello world”，source[1]是“a b c”，
        # st.values： ['hello', 'world', 'a', 'b', 'c']
        split_line = tf.string_split([line]).values

        # we load only one image for test, except if we trained a stereo model
        # 只读取一张图片做测试,除非训练一个立体模型
        if mode == 'test' and self.params.do_stereo:
            # 左边图片路径, ["hello", "world"] ==> "helloworld"
            left_image_path  = tf.string_join([self.data_path, split_line[0]])
            # 读取左边图片,返回resize后的图片
            left_image_o  = self.read_image(left_image_path)
        else:
            # 左右图片的路径
            left_image_path  = tf.string_join([self.data_path, split_line[0]])
            right_image_path = tf.string_join([self.data_path, split_line[1]])
            # 读取左右图片
            left_image_o  = self.read_image(left_image_path)
            right_image_o = self.read_image(right_image_path)

        # 训练模式
        if mode == 'train':
            # randomly flip images
            # 随机翻转图片

            # 根据生成的随机数判断图片是否进行左右翻转
            do_flip = tf.random_uniform([], 0, 1)
            left_image  = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_image_o), lambda: left_image_o)
            right_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_image_o),  lambda: right_image_o)

            # randomly augment images
            # 随机修改图片
            do_augment = tf.random_uniform([], 0, 1)
            left_image, right_image = tf.cond(do_augment > 0.5, lambda: self.augment_image_pair(left_image, right_image), lambda: (left_image, right_image))

            # 图片set shape
            left_image.set_shape([None, None, 3])
            right_image.set_shape([None, None, 3])

            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            # capacity: An integer.The maximum number of elements in the queue.
            # 容量: 一个整数, 队列中的最大的元素数
            min_after_dequeue = 2048
            capacity = min_after_dequeue + 4 * params.batch_size
            # 读取一个文件并且加载一个张量中的batch_size行
            # 从[left_image, right_image]利用 params.num_threads 个线程读取 params.batch_size 行
            # min_after_dequeue:当一次出列操作完成后, 队列中元素的最小数量, 往往用于定义元素的混合级别
            self.left_image_batch, self.right_image_batch = tf.train.shuffle_batch([left_image, right_image],
                        params.batch_size, capacity, min_after_dequeue, params.num_threads)

        elif mode == 'test':
            self.left_image_batch = tf.stack([left_image_o,  tf.image.flip_left_right(left_image_o)],  0)
            self.left_image_batch.set_shape( [2, None, None, 3])

            if self.params.do_stereo:
                self.right_image_batch = tf.stack([right_image_o,  tf.image.flip_left_right(right_image_o)],  0)
                self.right_image_batch.set_shape( [2, None, None, 3])

    def augment_image_pair(self, left_image, right_image):
        # randomly shift gamma
        # 随机移动
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug  = left_image  ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        # 随机改变亮度
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug  =  left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        # 随机改变颜色
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        # 新建一个 shape 为[tf.shape(left_image)[0], tf.shape(left_image)[1]],内容全是1的tensor
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        # 先将 white 随机化,然后按照 axis=2 进行拼接
        # Q: white 只有二维,axis=2 没有理解
        # A: [white * random_colors[i] for i in range(3)] 已经将之变为了三维,
        # axis=2 就是新增一阶张量,white:[tf.shape(left_image)[0], tf.shape(left_image)[1]],
        # 变为[tf.shape(left_image)[0], tf.shape(left_image)[1], 3]
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        # 随机改变颜色
        left_image_aug  *= color_image
        right_image_aug *= color_image

        # saturate
        # tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。
        # 小于min的让它等于min，大于max的元素的值等于max
        left_image_aug  = tf.clip_by_value(left_image_aug,  0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        # Q: [0] 不知道什么意思
        # 获取了图片路径的长度, 通过 python 的 len 函数
        path_length = string_length_tf(image_path)[0]
        # 获取图片路径的后缀名,str
        # tf.substr: 从字符串的 Tensor 中返回子字符串
        file_extension = tf.substr(image_path, path_length-3, 3)
        # 判断后缀名是否为"jpg",bool
        file_cond = tf.equal(file_extension, 'jpg')

        # 读取 ==> 解码 ==> 转换格式 ==> resize
        # 根据 file_cond 的值,判断用 jpeg 解码还是 png 解码,显示需要 tf.image.convert_image_dtype() 转换一下格式
        # tf.cond:类似于三元组函数, ?:fn1:fn2
        image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        # 如果数据集是 cityscapes,裁剪最后的1/5 以移除汽车引擎盖
        if self.dataset == 'cityscapes':
            o_height    = tf.shape(image)[0]
            crop_height = (o_height * 4) // 5
            image  =  image[:crop_height,:,:]

        # 转换之前解码得到的 image 的格式, tf.float32
        image  = tf.image.convert_image_dtype(image,  tf.float32)
        # 将 image resize 为设定好的长宽
        # ResizeMethod.AREA --> 基于区域的图像插值算法，首先将原始低分辨率图像分割成不同区域，然后将插值点映射到低分辨率图像，
        # 判断其所属区域， 最后根据插值点的邻域像素设计不同的插值公式， 计算插值点的值。
        image  = tf.image.resize_images(image,  [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)

        return image
