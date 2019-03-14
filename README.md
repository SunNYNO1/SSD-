# SSD-Tensorflow实验遇到的问题
***
## 1-输入命令格式问题（其中的\$\{TRAIN_DIR\}是linux系统的命令，在windows下直接去掉 \$\{\} 即可）：
TRAIN_DIR=./logs/ssd_300_vgg_3
CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} 
***
## <font color=#FF0000> 2-报错：'utf-8' codec can't decode byte 0xff in position 0: invalid start byte；解决（将“r”替换为‘rb’，同理将‘w’换为‘wb’）：
fix:
pascalvoc_to_tfrecords.py
image_data = tf.gfile.FastGFile(filename, 'r').read()
change that to:
image_data = tf.gfile.FastGFile(filename, 'rb').read()

### 2.1-为什么将"r"改为"rb"
对于读取的文件，utf-8解码出现错误（可能要用GBK解码，也可能有部分非utf-8编码内容，也可能是文件损坏），因此对于读取文件不再解码，直接二进制读入文件，即“rb”

### 2.2-补充（编码解码）：

GBK：采用双字节编码
GB18030：采用单字节、双字节、四字节分段编码（现在的中文信息处理应优先采用GB18030编码方案）
ASCII：是1个字节
Unicode：通常是2个字节。
UTF-8：把一个Unicode字符根据不同的数字大小编码成1-6个字节，英文字母被编码成1个字节，汉字是3个字节（ASCII编码实际上可以被看成是UTF-8编码的一部分）

在计算机内存中，统一使用Unicode编码，当需要保存到硬盘或者需要传输的时候，就转换为UTF-8编码

硬盘中的保存格式为byte类型，内存中为str格式
硬盘中用utf-8或ASCII编码，内存中用unicode编码

当从python（内存）中保存到硬盘时，需要将str类型编码为bytes类型。
如果内容是中文，只能用encode（utf-8）或 encode（GBK）编码；如果是英文，可以使用encode（ASCII）或encode（UTF-8）亦或者是encode（GBK）都可以。

当从硬盘读到python（内存）中时，如果是中文，只能使用decode（utf-8）解码；若是英文，decode（utf-8）或者decode（ASCII），主要看之前反过程是用encode（utf-8）还是encode（ASCII）编码的。；若两者都不可以，尝试encode（Unicode）抑或是decode（gbk，ignore）

参考链接：https://blog.csdn.net/qq_18888869/article/details/82625343  
https://blog.csdn.net/u013555719/article/details/77991010
***

## 3-报错:TypeError: Can not convert a tuple into a Tensor or Operation.解决（使用以下函数）：
def flatten(x): 
         result = [] 
         for el in x: 
              if isinstance(el, tuple): 
                    result.extend(flatten(el))
              else: 
                    result.append(el) 
          return result
in eval_op=flatten(list(names_to_updates.values()))
***

## 4-报错： No data files found in ./VOCdevkit/VOC2007/voc_2007_test_*.tfrecord
tfrecords路径设定出现问题（在指定生成路径保存tfrecords之后，下一次命令的读取路径与之前不一致，应该将tfrecords移至指定读取路径）
***

## 5-在保存tfrecords时，报错： Failed to create a NewWriteableFile
自己创建tfrecords保存路径
***

## 6-报错：InvalidArgumentError (see above for traceback): Default MaxPoolingOp only supports NHWC on device type CPU
at train_ssd_network.py line 27:  
DATA_FORMAT = 'NCHW'  
modify to:  
DATA_FORMAT = 'NHWC'
***

## 7-报错：ValueError: Can't load save_path when it is None. 
将命令中的checkpoints_path./checkpoints/ssd_300_vgg.ckpt改为CHECKPOINT_PATH='./checkpoints/ssd_300_vgg.ckpt/ssd_300_vgg.ckpt'
***
## 8-报错：Failed to get convolution algorithm. This is probably because cuDNN failed to initialize
tensorflow-gpu1.12.0版本太高了，将版本将为tensorflow-gpu1.9.0即可
命令：
conda list：查看安装情况  
conda uninstall tensorflow-gpu:卸载高版本
conda install --channel https//conda.anaconda.org/anaconda tensorflow-gpu==1.9.0：安装低版本
***
##  <font color=#FF0000> 9-报错:Could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED
### 9.1-报错原因：
源代码中的config = tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)语句限制了GPU内存的分配
删除掉该段代码即可或者是更为以下代码：config = tf.ConfigProto() config.gpu_options.allow_growth = True  
### 9.2-补充：
 tf.ConfigProto()函数用来对session进行参数配置
 * 参数：
 log_device_placement设置为True：获取到 operations 和 Tensor 被指派到哪个设备  
 allow_soft-palcement设置为True：允许自动选择一个存在且可用的设备
 * 限制GPU资源使用：  
     * 限制使用率：  
     config = tf.configProto()  
     config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%显存  
     session = tf.Session(config=config)
     或者是
     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
     config=tf.ConfigProto(gpu_options=gpu_options)
     session = tf.Session(config=config)

     * 动态分配内存：  
     config = tf.ConfigProto()  
     config.gpu_options.allow_growth = True  
     session = tf.Session(config=config)

     * 设置使用哪块GPU
     在python程序中设置：os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1  
     在执行时设置：CUDA_VISIBLE_DEVICES=0,1 python yourcode.py  
 * 参考链接：https://blog.csdn.net/dcrmg/article/details/79091941
***
## markdown使用参考链接：https://blog.csdn.net/qcx321/article/details/53780672
