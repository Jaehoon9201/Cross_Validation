# Keras Model Cross-Validation uisng a Kfold 
This code is for the Cross Validation using a Kfold.

Sample data is made for simple running this code.

Sample data is as follows.

```python
X= np.array([[1, 0, 0.8, 1] ,  [0.8, 0, 1, 1]  , [1, 0, 1, 0.8  ], [1, 0, 0.8, 0.8] , [0.8, 0, 0.8, 0.8],
            [1.1, 0, 1.3, 1],  [1.4, 0, 1, 1]  , [1, 0, 1.3, 1]  , [1, 0, 1.2, 1]   , [1.1, 0, 1.1, 1],
            [0, 0, 1, 1]    ,  [0, 0, 1, 1]    , [0, 0, 1, 1]    , [0, 0, 1, 1]     , [0, 0, 1, 1],
            [0, 0, 1.3, 1]  , [0, 0, 1, 1]     , [0, 0, 1.3, 1]  , [0, 0, 1.2, 1]   , [0, 0, 1.1, 1],
            [0.7, 0, 1, 1]  , [0.7, 0, 0.7, 1] , [0.5, 0, 1, 1]  , [0.5, 0, 0.5, 1] , [0.6, 0, 1, 0.5],
            [0, 0, 1.6, 1]  , [0, 0, 1, 1.6]   , [0, 0, 1.3, 1.6], [0, 0, 1.2, 1.6] , [0, 0, 1.1, 1.6],
            [1.7, 0, 1, 1]  , [1.7, 0, 0.7, 1] , [1.5, 0, 1, 1]  , [1.5, 0, 0.5, 1] , [1.6, 0, 1, 0.5],
            [0, 0, 1.6, 1.2], [0, 0, 1, 1.8]   , [0, 0, 1.3, 1.8], [0, 0, 1.2, 1.8] , [0, 0, 1.4, 1.6]])
y= [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
    [1],[1],[1],[1],[1],[1],[1],[1],[1],[1],
    [0],[0],[0],[0],[0],[1],[1],[1],[1],[1],
    [0],[0],[0],[0],[0],[1],[1],[1],[1],[1]]

```
There is a two kind of method.

## Method 1 : Cross_Validation_w_KerasModel_SimpleEx.py
It also draws the history of the learning process as processing epochs.
You can obtain the graph like below one if u are running this code.

![image](https://user-images.githubusercontent.com/71545160/124382332-dc8f0b80-dd01-11eb-902d-8600410c88cc.png)


## Method 2 :Cross_Validation_w_KerasModel_SimpleEx_ver2.py
This code is simple code for Cross Validation with KerasModel.

You can obtain the results like below one if u are running this code.
![image](https://user-images.githubusercontent.com/71545160/124382375-16601200-dd02-11eb-8d51-95a27e51f157.png)
