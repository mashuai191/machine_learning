1. Dataset is private in my own Kaggle private dataset.
2. one notebook is for buiding the model and the other one is for predicting using the pre-trained model
3. one useful point here is that one wedding image may have multiple styles, for example, '中国风'，'森系', '西式', and therefor this is not suitable for Softmax classification. We call this multi-label classification.
4. got many insight from https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/
5. multi-label classfication has a problem, for exapmle, if there is only 'red shirt', 'blue jeans' in training set, it's unlikely to predict 'blue shirt' because the network haven't seen that in training phase.
