# detection_tools
Some Useful Tools for Object Detection in Remote Sensing Imagery. 一些遥感图像目标检测的实用代码及工具，不断更新中...

## Usage 

* [ImgSplit_multiprocess_HBB.py](https://github.com/fan0210/detection_tools/blob/master/ImgSplit_multiprocess_HBB.py)
  
  对原始遥感图像训练数据进行裁切，生成固定大小的patches，适用于HBB(Horizontal Bounding Box)。

* [data_aug.py](https://github.com/fan0210/detection_tools/blob/master/data_aug.py)

  遥感图像目标检测数据扩充（在线增强），包括随机旋转、随机裁切、水平竖直翻转、随机线性增强、随机灰度化等，适用于OBB和HBB
