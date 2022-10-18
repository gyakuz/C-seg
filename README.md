# instance_segmentation
c++实现的一个实例分割深度学习模型，使用的模型是Mask RCNN (Inception v2 2018在COCO课程上训练)。使用OpenCV DNN模块，加载已经训练和导出的模型中，然后对输出进行后处理，在每个类实例周围绘制边界框，并对掩码进行着色。

## 依赖环境
-  C++ compiler for c++11 or later. \
-  OpenCV and opencv_conrib modules for c++ (OpenCV 4 version 4.5.5 was used, 4.0.0 or later required)

## 安装
下载并安装OpenCV库，操作系统是基于Unix的，在配置cmake recipe时，声明OpenCV贡献模块源目录的路径。它可以用cmake gui工具或终端命令完成，如下所示:
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/Alex/codes/aux_libs/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..
```

## 编译
为了成功编译segment.cpp，可能需要在每个opencv_contrib模块中包含一个标志作为编译器参数。＼

用于编译代码的命令是:
```
g++ -I/usr/local/include/opencv4/ -L/usr/local/lib/ segmentation.cpp  -lopencv_dnn -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -o segmentation.out
```


## 执行
程序接受图像。选择输入文件为 ```--image=/path/to/image. extension``` 在编译对象的名称之后。。
在Ubuntu终端上，执行编译的输出如下:
```
./segmentation.out --image=test.jpg
```

## 额外
使用的模型可以从tensorflow.com下载，使用:
```
wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz 
tar zxvf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

# ONNX部署
文件夹tensorflow-onnx将模型打包成ONNX格式
使用tf2onnx转换TensorFlow模型到ONNX

## 依赖环境
- 支持 tf-1.x graphs and tf-2.x. \
- 支持Python 3.7-3.10

## 安装
从pypi安装
```
pip install -U tf2onnx
```
从github安装最新版本
```
pip clone git+https://github.com/onnx/tensorflow-onnx
```
```
python setup.py install
```
## 运行
开始使用tensorflow-onnx，运行t2onnx.convert:
转换的模型是saved_mode格式\
命令：
```
python -m tf2onnx.convert --saved-model saved_model_dir --output model.onnx
```

示例：
```
python -m tf2onnx.convert --saved-model ./output/model  --output ./output/saved_mode.onnx
```

graphdef格式\
命令：
```
python -m tf2onnx.convert --input frozen_graph.pb  --inputs X:0 --outputs output:0 --output model.onnx
```

示例：
```
python -m tf2onnx.convert --input ./output/model/frozen_inference_graph.pb  --inputs Const:0,image_tensor:0 --outputs detection_masks:0 --output ./output/frozen_inference_graph.onnx
```

## 输出结果
output文件中放的是转换前.pd格式模式和转换后的.onnx文件，其中frozen_inference_graph.pb和saved_model.pb是instance segmentation model的俩种不同的保存格式
