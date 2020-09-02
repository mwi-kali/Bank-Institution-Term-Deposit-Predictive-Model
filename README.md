# Bank-Institution-Term-Deposit-Predictive-Model
## Introduction
The investment and portfolio department of a bank would want to be able to identify their customers who potentially would subscribe to their term deposits. As there has been heightened interest of marketing managers to carefully tune their directed campaigns to the rigorous selection of contacts.

## Objective
The goal is to find a model that can predict which future clients who would subscribe to their term deposit. Having such an effective predictive model can help increase the banks campaign efficiency as they would be able to identify customers who would subscribe to their term deposit and thereby direct their marketing efforts to them. This would help them better manage their resources.

## Data
The dataset was downloaded from the UCI ML website and more details about the data can be read from the same website.

## Data Pre-processing
### Encoding categorical variables
For better processing, categorical variables need to be encoded. This is through a label encoder to transform them into numerical columns. It encodes target labels with value between 0 and n_classes-1. This approach is very simple and it involves converting each value in a column to a number.
### Handling outliers
Outliers can adversely affect the training process of a machine learning algorithm, resulting in a loss of accuracy. To avoid this the outliers are removed by replacing it with central measures of tendencies.
### Scaling all numerical columns
This is through StandardScaler(). It standardizes features by removing the mean and scaling to unit variance.
### Dimensionality Reductions Techniques.
#### TSNE (t-distributed Stochastic Neighbor Embedding)
It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data.
#### Autoencoders
Autoencoder is an unsupervised artificial neural network. Its procedure starts compressing the original data into a short-code ignoring noise. Then, the algorithm uncompresses that code to generate an image as close as possible to the original input.
#### Principal Component Analysis
Principal Component Analysis (PCA) is a linear dimensionality reduction technique that can be utilized for extracting information from a high-dimensional space by projecting it into a lower-dimensional subspace. It tries to preserve the essential parts that have more variation of the data and remove the non-essential parts with fewer variations.
The dimensionality reduction technique chosen is Autoencoders. PCA features are totally linearly uncorrelated with each other since features are projections onto the orthogonal basis. Autoencoded features might have correlations since they are just trained for accurate reconstruction. The autoencoder output will be able to compress the information better into low dimensional latent space leveraging its capability to model complex nonlinear functions.

## Data Models
To select the model, cross validation to select the best machine learning models.Cross validation techniques used are Stratified K-fold and K-fold. Stratified K-fold is the chosen cross validation technique.
Evaluation metrics used are ROC, F1 Score, Accuracy, Precision and Recall.
AUC score 1 represents a perfect classifier, and 0.5 represents a worthless classifier. F1 score is the amount of data tested for the predictions. Accuracy is the subset accuracy. The set of labels predicted for a sample must exactly match the corresponding set of labels in y_true. Precision score means the level up-to which the prediction made by the model is precise. Recall is the amount up-to which the model can predict the outcome.

### Logistic Regression Model
Logistic regression is a statistical method for predicting binary classes. The outcome or target variable is dichotomous in nature. Dichotomous means there are only two possible classes.
AUC score for the case is 0.7894.
The mean Cross-Validation Accuracy is 0.8916.
F1 Score: 0.2632
Accuracy: 0.8844
Precision: 0.6028
Recall: 0.1683

### XGBoost
Boosting is a sequential technique which works on the principle of an ensemble. It combines a set of weak learners and delivers improved prediction accuracy. At any instant t, the model outcomes are weighed based on the outcomes of previous instant t-1. The outcomes predicted correctly are given a lower weight and the ones miss-classified are weighted higher.
AUC score for the case is 0.7927.
The mean Cross-Validation Accuracy is 0.8916
F1 Score: 0.2621
Accuracy: 0.8852
Precision: 0.6176
Recall: 0.1663

### Multilayer Perceptron
A multilayer perceptron or MLP represents a vast artificial neural network, meaning simply that it features more than one perceptron. This gathering of perceptrons is established from an input layer meant to receive the signal, an output layer responsible for a decision or prediction in regards to the input, and an arbitrary number of hidden layers that represent the true computational power of the MLP.
AUC score for the case is 0.7102.
The mean Cross-Validation Accuracy is 0.8918.
F1 Score: 0.3166
Accuracy: 0.5419
Precision: 0.1937
Recall: 0.8653

### Support Vector Machine
SVM offers very high accuracy compared to other classifiers such as logistic regression, and decision trees. It is known for its kernel trick to handle nonlinear input spaces. It is used in a variety of applications such as face detection, intrusion detection, classification of emails, news articles and web pages, classification of genes, and handwriting recognition.
AUC score for the case is 0.7462.
The mean Cross-Validation Accuracy is 0.8885.
F1 Score: 0.0000
Accuracy: 0.8774
Precision: 0.0000
Recall: 0.0000

### Decision Trees
A decision tree is a flowchart-like tree structure where an internal node represents feature(or attribute), the branch represents a decision rule, and each leaf node represents the outcome. The topmost node in a decision tree is known as the root node. It learns to partition on the basis of the attribute value. It partitions the tree in recursively manner call recursive partitioning. This flowchart-like structure helps you in decision making. It’s visualization like a flowchart diagram which easily mimics the human level thinking. That is why decision trees are easy to understand and interpret.
AUC score for the case is 0.7891.
The mean Cross-Validation Accuracy is 0.8917.
F1 Score: 0.2609
Accuracy: 0.8844
Precision: 0.6043
Recall: 0.1663

### Random Forests
Random forests is a supervised learning algorithm. It can be used both for classification and regression. It is also the most flexible and easy to use algorithm.
A forest consists of trees. It is said that the more trees it has, the more robust a forest is. Random forests create decision trees on randomly selected data samples, get prediction from each tree and select the best solution by means of voting. It also provides a pretty good indicator of the feature importance.
AUC score for the case is 0.7883.
The mean Cross-Validation Accuracy is 0.8906
F1 Score: 0.2646
Accuracy: 0.8840
Precision: 0.5931
Recall: 0.1703

## Conclusion
The model has been chosen based on the accuracy, the interpretability, the complexity and the scalability of the model.
The best model would be XGBoost.
The mean Cross-Validation Accuracy is 0.8942.
The AUC score for the XGBoost is 0.7927 this is closer to 1 than all the other AUC scores. The F1 Score is 0.2621. The accuracy is 0.8852. The precision is 0.6176. The recall is 0.1663.
The next best model would be tie between Logistic Regression and Multi-layer Perceptron.
Logistic Regression
The mean Cross-Validation Accuracy is 0.8941
The AUC score for the Logistic Regression is 0.7894. The F1 Score is 0.2632. The accuracy is 0.8844. The precision is 0.6028. The recall is 0.1683.
Multi-layer Perceptron
The mean Cross-Validation Accuracy is 0.8941.
The AUC score for the Multi-layer Perceptron is 0.7102. The F1 Score is 0.3166. The accuracy is 0.5419. The precision is 0.1937. The recall is 0.8653.
