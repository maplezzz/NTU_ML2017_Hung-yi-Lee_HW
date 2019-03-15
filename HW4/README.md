# HW4 Text Sentiment Classification    
[HomeWork description](https://docs.google.com/presentation/d/1NztROuZdlCUlB7xNPLvJ7cMZsy2GR_d8UBEoJHT7eSc/edit?usp=sharing)  
[dataset](https://drive.google.com/file/d/1LvonIrqb7UnrWu9h2zlR0Co2bssVZMUN/view?usp=sharing)  

## Requirement  
 
- Processing the Sentence  

    先建立字典，字典內含有每一個字所對應到的index

    利用Word Embedding來代表每一個單字，並藉由RNN model 得到一個代表該句的vector

- Word Embedding  

    用一些方法pretrain 出word embedding (ex：skip-gram、CBOW )

- Semi-Supervised learning  
  
    semi-supervised 簡單來說就是讓機器自己從unlabel data中找出label，而方法有很多種，這邊簡單介紹其中一種比較好實作的方法 Self-Training

    Self-Training 把train好的model對unlabel data做預測，並將這些預測後的值轉成該筆unlabel data的label，並加入這些新的data做training。

- 請使用RNN實作model
 



  





