# HW3 Image Sentiment Classification    
[HomeWork description](https://ntumlta.github.io/ML-Assignment3/index.html)  

由于dataset过大，就不放在github上

##Requirement  
 
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

##Data description  

本次作業為Image Sentiment Classification。我們提供給各位的training dataset為兩萬八千張左右48x48 pixel的圖片，以及每一張圖片的表情label（注意：每張圖片都會唯一屬於一種表情）。總共有七種可能的表情（0：生氣, 1：厭惡, 2：恐懼, 3：高興, 4：難過, 5：驚訝, 6：中立(難以區分為前六種的表情))。  
![](https://github.com/maplezzz/ML2017S_Hung-yi-Lee_HW/blob/master/HW3/img/data_example.png)

Testing data則是七千張左右48x48的圖片，希望各位同學能利用training dataset訓練一個CNN model，預測出每張圖片的表情label（同樣地，為0~6中的某一個）並存在csv檔中。