# HW3 Image Sentiment Classification    
[HomeWork description](https://ntumlta.github.io/ML-Assignment3/index.html)  
[PPT](https://docs.google.com/presentation/d/1txLnBXLYmpJOMsDJItB81lA1gHJ21fgnZTxXWZtgryE/edit?usp=sharing)  
[dataset](https://drive.google.com/file/d/1UGM_CJkNb7OmUQKpxSmaUETiCQd_OBus/view?usp=sharing)  

## Requirement  
 
- Build Convolution Neural Network  

    Build CNN model, and tune it to the best formance as possible as you can.

    Record your model structure and training procedure.

- Build Deep Neural Network  

    Using the same number of parameters as above CNN, build a DNN model to do this task.

    Record your model structure and training procedure. Explain what you observed.

- Analyze the Model by Confusion Matrix  

    Observe the prediction of your validation data( 10% ~ 20% of training data is OK ).

    Plot the prediction into confusion matrix and describe what you observed.

- Analyze the Model by Plotting the Saliency Map
 
    Plot the saliency map of original image to see which part is important when classifying

- Analyze the Model by Visualizing Filters (1%)

    Use Gradient Ascent method mentioned in class to find the image that activates the selected filter the most and plot them.
    
**[Visualization Tutorial](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)**    

## Data description  

本次作業為Image Sentiment Classification。我們提供給各位的training dataset為兩萬八千張左右48x48 pixel的圖片，以及每一張圖片的表情label（注意：每張圖片都會唯一屬於一種表情）。總共有七種可能的表情（0：生氣, 1：厭惡, 2：恐懼, 3：高興, 4：難過, 5：驚訝, 6：中立(難以區分為前六種的表情))。  
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/data_example.png)

Testing data則是七千張左右48x48的圖片，希望各位同學能利用training dataset訓練一個CNN model，預測出每張圖片的表情label（同樣地，為0~6中的某一個）並存在csv檔中。  
  
## Data analysis and clean  
  
使用pandas, numpy等工具将原始数据train.csv进行处理分析, 将feature以shape(48,48), 将label以一维np.array的格式输出, 具体方式详见[data_analysis.ipynb](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/data_analysis.ipynb)  
  
将7个类别的图片数量统计   

| angry       | disgust     | fear       | happy      | sad        | surprise   | neutral    |    
| ----------- | ----------- |----------- |----------- |----------- |----------- |----------- |  
| 3995        | 436         |4097        |7215        |4830        |3171        |4965        |


可见分类为厌恶的图片的数量明显偏少, 构建模型需练的时候需要调整weight以达到更好的效果  
  
以下是每个表情的实例图片  
  
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/data_analysis/angry.png)
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/data_analysis/disgust.png)  
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/data_analysis/fear.png)
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/data_analysis/happy.png)  
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/data_analysis/sad.png) 
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/data_analysis/surprise.png)
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/data_analysis/neutral.png)  
  
  
## Report  
  
### Build CNN    
  
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/cnn.png)  
  
CNN的结构如图所示, 主要有3个conv box以及在flatten后接一个dense_layer, 每一个conv box里面由convolution_layer, batch_normalization, max_pool, dropout构成, 所有Conv2D中kernel_size=5, strides=1, MaxPooling2D中pool_size=2, strides=1, 除了最后一层为softmax, 其他所有层的activation均为relu, Total params: 2,209,059, Trainable params: 2,207,893, Non-trainable params: 1,166  
  
训练过程中accuracy与loss变化如下图所示  
  
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/report/cnn_acc.png)
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/report/cnn_loss.png)   
    
### Build DNN  
  
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/dnn.png) 
  
DNN有三层Dense_layer构成, 每层的数目分别为512, 1024, 512, 每层后面均有batch_normalization和dropout, activation同样为relu,Total params: 2,242,083, Trainable params: 2,237,973, Non-trainable params: 4,110 训练过程中accuracy与loss变化如下  
  
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/report/dnn_acc.png)
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/report/dnn_loss.png)  

#### 比较  
在可训练参数数目相差不多的情况下, dnn每一个epoch的训练时间约为4s, 远快于cnn的19s, 但是在同样做了BN以及droupout的情况下, cnn的训练效果远好于dnn.  

在验证集为数据集最后2000笔的情况下, cnn在验证集的正确率达到了60%, 而dnn只有40%, 证明了课上的理论, cnn能够更高效的利用每一个参数  

### Analyze the Model by Confusion Matrix  

测试资料来源于训练资料的最后2000笔, 大约10%的训练资料, 下图左边为CNN的Confusion Matrix, 右边为DNN的, 可见CNN的训练效果远好于DNN  

![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/report/cnn_cm.png)
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/report/dnn_cm.png)  

这里主要拿CNN的Confusion Matrix做主要分析, 除去过少的分类为厌恶的图片, 分类为开心的图片训练效果最好, 分类为伤心的图片训练效果最差, 其中很大一部分被错分为中立, 下图为9张被错分的图片  

![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/report/sad_n.png)  

可见确实有几张即便通过人类也很难正确做出和训练集一样的分类答案, 具体操作方式见[report.ipynb](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/report.ipynb)  

### Analyze the Model by Plotting the Saliency Map  

从[keras](https://raghakot.github.io/keras-vis/visualizations/saliency/)关于Saliency Map的教程可以知道, Saliency Map其实就是最后的结果对输入的求导,从而判断原图中每个像素对最后结果的影响.  

下图为CNN模型对测试集前6张图片的Saliency Map, 具体操作方式见[report.ipynb](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/report.ipynb), 需要注意的是由于keras的vis api与tensorflow的keras api有版本冲突, 所以如果之前训练的模型是用tf.python.keras训练并保存的话, 需要用keras重新训练保存, 并且用keras.models.load_model 读取储存的模型  

![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/report/sm0.png)
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/report/sm1.png)  
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/report/sm2.png)
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/report/sm3.png)  
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/report/sm4.png)
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/report/sm5.png)  

从Saliency Map可以看出CNN 模型对结果影响较大的像素点主要集中在原图的五官以及面部轮廓附近, 这样的结果是显而易见的

