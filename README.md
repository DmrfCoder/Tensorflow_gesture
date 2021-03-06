# Tensorflow_gesture
## API封装需求描述
将原有GestureWithNcnn项目中的发送声波信号、接收声波信号、signal ndk部分抽取出来做封装成新的工具类并打包成sdk放置到github或其他平台使得新的demo可以通过gradle链接到API，新的项目引用API后可直接调用对应的功能而无需关注这部分代码的实现。
## 如何将.csv文件中的数据解析并进行pc端预测

+ 使用Utils/ReadCsv.py文件读取并解析csv文件，执行完成后会将数据以txt文件的形式存至path路径
+ 将txt格式的文件切成0.5s的微手势
+ 使用切成的微手势利用Predict/gesture_cnn_pb_predict.py文件进行预测
## LSTM实现思路
![这里写图片描述](https://img-blog.csdn.net/20180424184731423?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTgyMTYw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

连续手势的持续时长有1s和2s两种持续时长，先将连续手势中的微手势（2*0.5s和4*0.5s）作为cnn的输入，这时一个连续手势通过cnn后可以产生2个或4和256维的输出，把这些256的输出格式化为等长（以最长的4*256为标准，将2*256中不4*256的部分补0），这样连续手势经过cnn后的输出都为4*256，将这4*256的一维向量作为LSTM的输入进行训练。

### 初步结果

![这里写图片描述](https://img-blog.csdn.net/20180501230141795?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTgyMTYw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
### Confustion Matrix
![这里写图片描述](https://img-blog.csdn.net/20180514093234491?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTgyMTYw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

对应的error_count（train）可视化结果：

![这里写图片描述](https://img-blog.csdn.net/20180514224436319?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTgyMTYw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

其中B手势和I手势的识别结果和预期出现了比较大的偏差，初步估计是label的问题，待修复。
## tensorflow->android demo需求说明
输入数据维度为2x8x550，pb文件是已有的，输出预测得到的label值。
## tensorboard 可视化需求说明
包括但不限于：train_loss&test_loss&train_accuracy&test_accuracy&mean of kernels&weights of kernels

