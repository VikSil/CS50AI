# Project 5 - Traffic

The goal of CS50AI course Project 5 "Traffic" is to develop and optimise a deep convolutional neural network for traffic sign recognition. This document describes one possible solution and steps to optimisation. 

## Initial setup

The general structure of a convolutional image processing neural network (hereafter - NN) consists of convolutional layer, followed by a pooling layer, followed by a flattening layer. The latter produces an array of outputs that can be passed into a traditional deep learning NN with multiple hidden layers.

The input shape and the output shape are given in the project specification as `(IMG_WIDTH, IMG_HEIGHT, 3)` for the input shape and `NUM_CATEGORIES` for the output shape. Hence, the minimal NN setup for the given specification can be described by the following code:

        model = tf.keras.models.Sequential(
        [
           # input layer
           tf.keras.layers.Conv2D(
               # minimal number of filters = 1  
               # smallest possible kernel on a filter =  2 x 2
               1, (2,2), activation = 'relu', input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
               ), 
           
           # pooling layer reduces 2 x 2 regions to single max value
           tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
           
           tf.keras.layers.Flatten(),
           
           # hidden layer with one node
           tf.keras.layers.Dense(43,activation = 'relu'),
           
           # no dropout
           tf.keras.layers.Dropout(0.0),
           
           tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
        ])

Running `print(model.summary())` for the above model displays summary:

![Initial NN summary](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/01.png)


## Parameters

Several parameters of the initial NN setup can be easily changed in order to improve performance. These parameters (with feasable value ranges in brackets) are:

* Number of filters on the convolutional layer (1 to 100).
* Size of the filters on the convolutional layer ((2 x 2) to (5 x 5)).
* Size of the pooling kernel ((2 x 2) to (5 x 5)).
* Type of the pooling layer (max or average).
* Number of nodes in the hidden layer (14 to 260).
* Amount of dropout (0% to 70%).

## Methodology

Testing each combination of all possible values for every setup parameter is not practically feasable. Small changes in one or two parameters are unlikely to yield a notable differences in the NN performance. Therefore, rather than iterrating through each possible NN setup, a random sample of NN setups were tested and performance metrics were recorded. The sampling code is available in [`traffic_sampling.py`](https://github.com/VikSil/CS50AI/blob/trunk/Project5/traffic/traffic_sampling.py). The sampling results are available in [`sample.csv`](https://github.com/VikSil/CS50AI/blob/trunk/Project5/traffic/sample.csv).

## Performance metrics

Each NN setup was trained for 10 epochs, followed by a validation with a distinct dataset. Perfomance of each NN setup was evaluated on the following criteria:
* Accuracy on the validation set - higher accuracy is better.
* Loss on the validation set - lower loss is better.
* Number of epochs to achieve accuracy within 1% of the validation accuracy - less epochs is better.
* Whether validation set has higher accuracy than last training set.
* Whether validation set has lower loss than last training set.
* Total number of trainable parameters - less parameters is better, all other metrics being equal.

For reference, performance of the inital minimal NN:

Metric |Value|
--|--|
Accuracy | 0.6000 
Loss | 1.3029
Epochs to accuracy convergence | 10
Validation set has highest accuracy | True
Validation set has lowest loss | False
Trainable parameters | 296


## Analysis 

The sample set was loaded into a Pandas Dataframe, sorted and visualised with Matplotlib. The code for analysis is available in [`analysis.py`](https://github.com/VikSil/CS50AI/blob/trunk/Project5/traffic/analysis.py).


The sample set was evaluated and iterratively pruned, for each metric removing setups that did not yield a high output. The dataset was pruned as follows:

### Validation Accuracy

Validation accuracy in the sample ranged from 0.0575 to 0.9877. Binning accuracy values accross all runs into ten bins showed that around 2000 runs had validation accuracy higher than 90%.

![Binned Frequency of Validation Accuracy](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/02.png)

Further analysis showed that the cut-off for the upper 10th percentile was at 97.61% of accuracy. There were 302 runs with validation accuracy equal or higher than 97.61%. These runs were preserved in the sample set for further analysis, and the remaining runs were removed.

### Validation Loss

Validation accuracy in the pruned sample ranged from 0.0369 to 0.1334.
Binning loss values accross all runs into eight bins showed that majority of the runs had loss less than 0.1.

![Binned Frequency of Validation Loss](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/03.png)

Further analysis showed a correlation between accuracy and loss - higher accuracy had a tendency to coincide with lower loss.

![Relationship between Validation Accuracy and Loss](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/04.png)

This means that less accurate runs could be removed from the sample set without fear of removing runs with high accuracy. A total of 113 runs with loss higher or equal to 1.0 were pruned from the sample, leaving 189 runs.

### Number of Epochs till Convergence

The analysis showed that majority of remaining runs were never within 1% of the validation accuracy over the 10 training epochs (zero epochs to convergence). 
![Frequency of Epochs till Convergence](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/05.png)

Convergence had no clear correlation with either validation accuracy or validation loss, however runs with higer accuracy and lower loss tended to not converge more frequently.

![Relationship between Validation Accuracy and Convergence](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/06.png)

![Relationship between Validation Accuracy and Loss](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/07.png)

In order to reduce the sample set two additional columns were introduced, ranking each run by its validation accuracy and validation loss. Dataset was then ordered by the product of the newly introduced rows. The first 100 rows with the lowest rank and zero convergence were removed from the dataset.

After the pruning the lower bound of valuation accuracy in the sample had changed from 0.9762 to 0.9793. The upper bound of valuation loss had changed from 0.0998 to 0.094 


### Increased Validation Accuracy and Decreased Validation Loss

Each epoch of training should increase the accuracy of the NN, i.e. the accuracy of the second epoch should be higher than first, accuracy of the third epoch highter than second, and so on. A well performing NN should be equally performant on a validation set as it is on a training set. Thus, it stands to reason that after ten epochs the 11th epoch should have even higher accuracy than the 10th epoch. Likewise, loss should decrease with each epoch and, ideally, should be less for the validation set than the 10th epoch of the training set.

Analysis revealed that indeed majority of the runs remaining in the sample had higher validation accuracy and lower validation loss than the 10th epoch of the training set.

![Percentage of Increased Valuation Accuracy](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/08.png)

![Percentage of Decreased Valuation Loss](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/09.png)

The sample set was pruned by removing all runs that did not have either increased validation accuracy or decreased validation loss. After pruning 67 runs remained in the sample set.


### Number of Trainable Parameters

Trainable parameters are the number of weights that can be adjusted in the process of NN training. Since NNs with more weights take longer to train, given equal performance on other parameters, a NN with less trainable parameters is favourable. 

Number of trainable paramters in the pruned sample ranged from 181 103 to 3 913 822. Binning the number of trainable variables accross all runs into ten bins showed that most runs fall within the first two bins, and more runs fall within the first bin than in any other bin.

![Binned Frequency of Number of Trainable Parameters](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/10.png)

Plotting validation accuracy and loss against the number of bins revealed that indeed high accuracy and low loss can be achieved by a relatively small number of trainable parameters. 

The sample set was pruned by removing all runs with more than 1.2 million trainable parameters. Afterwards, there were 45 runs remaining in the sample. 

### Setup Analysis

It may be interesting to take a look at the setup of the remaining 45 best runs in the sample:

* Number of convolutional filters ranged from 19 to 99 with 39-59 filters being used most frequently

![Binned Frequency of Numbers of Filters](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/13.png)

* Filter size `(5 x 5)` appears to be strongly favourable for performance. 66.7% of the 45 best runs used filter size `(5 x 5)`, with remaining 33.3% using filter size `(4 x 4)`. Other sampled filter sizes were not present in the pruned set of 45 best runs.

![Filter Size](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/14.png)

* Middle sized pooling kernels appears to be more favourable for performance than small or large kernels. 58.3% of the 45 best runs used pooling kernel size `(3 x 3)`, followed by kernel size `(4 x 4)` used on 35.4% runs. Less than 10% combined used the smallest kernel `(2 x 2)` or the largest kernel `(5 x 5)`. 

![Pooling Kernel Size](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/15.png)

* All best performing runs used max pooling
* Number of hidden nodes ranged from 107 to 260, with 184 to 221 nodes used most frequently.

![Binned Frequency of Numbers of Hidden Nodes](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/16.png)

* Dropout ranged from 11% to 59%, with 23%-35% dropout occuring most frequently.

![Binned Frequency of Dropout](https://raw.githubusercontent.com/VikSil/CS50AI/refs/heads/trunk/Project5/traffic/img/17.png)


### Final pruning

The remaining best runs were further narrowed down by comparing performance metrics. Validation accuracy, validation loss and number of trainable parameters are the relevant metrics, since they have the highest variability. Each of the remaining runs was compared to other runs and discarded, if there was any other run in the set with simultaneously higher validation accuracy, lower validation loss and lower number of trainable parameters. After pruning the following eleven runs remained in the sample. 

|index|accuracy|loss|acc_convergence|parameters|hidden_nodes|filters|filter_size|pooling_size|dropout|accuracy_rank|loss_rank|rank|
|--|--|--|--|--|--|--|--|--|--|--|--|--|
1547|0.9877|0.0658|0|913867|190|74|(5x5)|(3x3)|31|1|3|3
1753|0.9863|0.0639|0|767147|218|54|(5x5)|(3x3)|34|3|1|3
806|0.9876|0.0654|0|651039|194|91|(5x5)|(4x4)|35|2|2|4
705|0.9858|0.0735|0|606495|166|99|(5x5)|(4x4)|31|5|12|60
2691|0.9849|0.073|10|526836|250|57|(4x4)|(4x4)|21|10.5|11|115.5
764|0.9849|0.077|0|501959|113|68|(5x5)|(3x3)|31|10.5|16|168
2917|0.984|0.0722|0|338323|142|64|(5x5)|(4x4)|25|18|10|180
2199|0.9845|0.074|0|379235|260|22|(5x5)|(3x3)|49|14|13|182
1494|0.9847|0.0786|0|337775|215|42|(5x5)|(4x4)|26|13|19.5|253.5
1590|0.9848|0.085|0|194841|172|30|(4x4)|(4x4)|24|12|61|732
2852|0.9812|0.0888|0|181103|235|20|(5x5)|(4x4)|38|69|92|6348

At a glance it is obvious that there are three high ranking setups at the top of the list. Of these the third (index 806) apears to be the optimal one. Validation accuracy and validation loss are ranked second, and this setup has significanlty less trainable parameters while achieving close to the same performance as the runs that are ranked highest by accuracy and loss.

## Conclusion

The goal of this experiment was to find the best performing setup of a deep convolutional image processing neural network. In order to achieve this goal 3000 runs of comparable NN setups were sampled. Extensive analysis was carried out on the resulting sample data. 

Notable observations that were made about the sample data:
* Two thirds of all samples were within 90% of accuracy. This implies that overall architecture of the NN is significantly more important in achieving performance than particular size or number of elements in each layer.
* Validation accuracy appears to have inverse relationship to validation loss - higher accuracy tends to correlate with lower validation loss.
* For most performing samples training accuracy did not come within 1% of the validation accuracy over 10 epochs. This suggests that increasing the number of traing epochs could further improve performance.
* For performant setups, more ofthen than not higher accuracy and lower loss was achieved on the validation set than on any of the training sets.
* Increasing the number of trainable parameters above approximatelly 1.2M has deminishing returns on the performance.
* Larger filter layer kernel size is more favourable, however it is not clear where the upper bound is, since kernels larger than (5 x 5) were not sampled.
* Middle sized pooling layer kernels yield better performance than very large or very small kernels.
* Max pooling setups significantly outperforms average pooling setups - all higest performing setups use max pooling.
* There is no clear best setup for the number of nodes in the hidden layer. It may be that the number is significant in relation to other parameters rather than on its own.
* Majority of best performing setups employ dropout between 23% and 47%.

The best performing setup was identified in the sample set having the following parameters:

Parameter |Value|
--|--|
Number of filters | 91
Filter kernel size | (5 x 5)
Pooling kernel size | (4 x 4)
Number of nodes in hidden layer | 194
Dropout | 35%


This setup acchieved the following perfomance:

Metric |Value|
--|--|
Accuracy | 0.9876 
Loss | 0.0654
Epochs to accuracy convergence | No convergence
Validation set has highest accuracy | True
Validation set has lowest loss | True
Trainable parameters | 651 039

However, it is important to note that performance of a NN on validation set is partially dependant on luck inherent in the randomness of choosing the training sets. Results of this experiment could be further explored by sampling repeated runs of each of the best performing setups, and ultimetelly choosing the one that consistently shows the best results.