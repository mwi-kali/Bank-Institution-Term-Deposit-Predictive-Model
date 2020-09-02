# Bank-Institution-Term-Deposit-Predictive-Model
## Introduction
The investment and portfolio department of a bank would want to be able to identify their customers who potentially would subscribe to their term deposits.

## Objective
The aim of this activity is to find a machine learning model that can predict which future clients who would subscribe to their term deposit. This will increase the banks campaign efficiency and effectiveness.
The bank would be able to identify customers who would subscribe to their term deposit and use that knowledge to direct their marketing efforts to them. This is known as target marketing.
This would help them better manage their resources and avoid waste.

## Data
The datasets were downloaded from the UCI ML website and more details about the data can be read from the same website.

## Data Pre-processing
Encoding categorical variables
In order to process data accurately, categorical variables need to be encoded to numeric values. This can be done using different types of encoders.
In this activity, a label encoder is used to transform categorical variables into numerical variables. It encodes categorical variables with values between 0 and n_classes-1. N_classes is the number of categories.

### Handling outliers
Outliers are known to skew the data. Outliers affect the training process of a machine learning algorithm, resulting in a loss of accuracy. This can lead to an inaccurate model.
In order to get an accurate model, the outliers are removed by replacing it with central measures of tendencies. In this activity the measure used is mean.

### Scaling all numerical columns
Data sets with variables that have varied magnitudes, units and range affect the training process of a machine learning algorithm. This is because most of the algorithms utilized in machine learning use Euclidean distance between two data points in their computations. This gives differing results.
Standardization brings the variables to a common magnitudes, units and range. There many ways of standardizing data. In this activity, StandardScaler() is used. It standardizes features by removing the mean and scaling to unit variance.

### Dimensionality Reductions Techniques
These are techniques that estimate how informative each column is and to remove the columns that are not. In this activity, t-distributed stochastic neighbor embedding, autoencoders and principal component analysis are explored separately them compared to find the best dimensionality reductions technique to use.

#### TSNE
T-distributed Stochastic Neighbor Embedding (TSNE) reduces the reduces dimensions based on non-linear local relationships among the data points. It tries to minimize the Kullback-Leibler divergence between the joint probabilities of

#### Autoencoders
Autoencoder is an unsupervised artificial neural network trained with the back-propagation algorithm to reproduce the input vector onto the output layer. Its procedure starts compressing the original data into a short-code ignoring noise using an encoder. This is followed by an algorithm that decompresses the short-code to generate the data as close as possible to the original input using the decoder.

#### PCA
Principal Component Analysis (PCA) is a statistical process that perpendicularly transforms the original numeric columns of data into a new set of columns called principal components. It is a linear dimensionality reduction technique.
It can be utilized for extracting information from a high-dimensional space. It preserves the essential columns that have more variation of the data and remove the non-essential columns with fewer variations.

## Data Models
To select the model, cross validation to select the best machine learning models.Cross validation techniques used are Stratified K-fold and K-fold.
Stratified K-fold is the chosen cross validation technique. Using Stratified K- Fold because it generates test sets such that all contain the same distribution of classes, or as close as possible. It preserves order dependencies in the dataset ordering. The Stratified K- Fold has a small range of cross-validation accuracy scores.
Evaluation metrics used are:
AUC score 1 represents a perfect classifier, and 0.5 represents a worthless classifier.
F1 score is the amount of data tested for the predictions.
Accuracy is the subset accuracy. The set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
Precision score means the level up-to which the prediction made by the model is precise.
Recall is the amount up-to which the model can predict the outcome.
For the above, a class helper will contain the code:

### Logistic Regression Model
Logistic regression is a statistical method for predicting binary classes. The outcome or target variable are only two possible classes.

### XGBoost
Boosting is a sequential technique which combines a set of weak learners and delivers improved prediction accuracy. The outcomes predicted correctly are given a lower weight and the ones miss-classified are weighted higher.

### Multi-layer Perceptron
A multi-layer perceptron gathers perceptrons that are established from an input layer meant to receive the signal. It has an output layer responsible for a decision or prediction in regards to the input. Multi-layer perceptrons have an arbitrary number of hidden layers that represent the true computational power of the machine learning algorithm.

### Support Vector Machine
Support Vector Machine handles nonlinear input spaces. It uses classification algorithms for two-group classification problems.
It is used in a variety of applications such as face detection, intrusion detection, classification of emails, news articles and web pages, classification of genes, and handwriting recognition.

### Decision Trees
A decision tree is like a flowchart tree structure with an internal node, a branch and leaf nodes. The internal node represents feature. The branch represents a decision rule. Each leaf node represents the outcome. The root node is the highest node.
Decision trees learn to partition on the basis of the attribute value. It partitions the tree using recursive partitioning.
Decision trees helps in decision making. The visualization of decision trees is like a flowchart diagram. This makes it easy to understand and interpret for it mimics the human level thinking.

### Random Forests
A random forest consists of trees. The more trees, the more robust a forest is. The trees are decision trees on randomly selected data samples. This is to get prediction from each tree and select the best solution by means of voting.

## Conclusion
The model has been chosen based on the accuracy, the interpretability, the complexity and the scalability of the model.

The best model would be XGBoost.

The next best model would be tie between Logistic Regression and Multi-layer Perceptron.

Read more here https://medium.com/analytics-vidhya/bank-institution-term-deposit-predictive-model-14af2bbba70e?source=friends_link&sk=0506757df92e80d4e5b693b77f83d386 remember to leave a clap.
