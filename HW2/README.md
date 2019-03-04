# HW2 Incoming Prediction   
[HomeWork description](https://docs.google.com/presentation/d/13LSyr3XV4ZrUYimgzzyLn8l50HphTQtvJ5n7XZZCwTw/edit?usp=sharing)  
  
## Requirement  
**Dataset and Task Introduction**  
- TASK: Binary Classification   
   
    Dtermine whether a person makes over 50K a year   
    
- Dataset: ADULT   
    
    Extraction was done by Barry Becker from the 1994 Census database.     
    A set of reasonably clean records was extracted using the following conditions: ((AGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)).  
    
- [Reference](https://archive.ics.uci.edu/ml/datasets/Adult)   


**Data Attribute Information**  
- train.csv 、test.csv:   
    age, workclass, fnlwgt, education, education num, marital-status, occupation  
    relationship, race, sex, capital-gain, capital-loss, hours-per-week,  
    native-country  

- make over 50K a year or not  

- For more details please check out [Kaggle’s](https://www.kaggle.com/c/ml-2017fall-hw2) Description Page  

## Result  
**Probabilstic Generative Model**  

在这个模型里面假设数据集属于高斯分布，采用两个种类shared_sigma的模型，通过数据集算出各自的<img src="https://latex.codecogs.com/gif.latex?\mu&space;_{1},&space;\mu&space;_{2},&space;\Sigma" title="\mu _{1}, \mu _{2}, \Sigma" />，然后直接带入公式求解  
  
训练数据直接将其中的10%当作valid，最后valid accuracy为0.843366, test accuracy为0.843867，结果还是很不错的  

**Logistic Regression**  

实验过程中采取过Ada但是效果不如sgd，使用mini-batch加快速率，batch-size为32，一共进行300epoch，最终结果  
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW2/output/TrainProcess.png)  
随着epoch-time的增多，loss不断减小，最后valid accuracy为0.852858, test accuracy为0.852343，优于Probability Generative Model这是因为这个模型下不需要假设采样数据的分布  

**Neural Network**   

使用keras三层的fully-connected neural network，使用的loss是binary_crossentropy，activation='sigmoid'，optimizer='adam'，前两层都是600个units，batch-size=32， epoch-times=50  
最后在valid set上的acc=0.9084, test-set上的acc为0.8426

