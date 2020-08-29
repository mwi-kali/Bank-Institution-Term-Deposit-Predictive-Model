class helper:
    def __init__(self):
        print ("Helper object created")
    def confusion(self, y_pred):
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        class_names=[0,1] # name  of classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

        # create heatmap
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap=colour_palette ,fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

    def roc_plot(self,model):
        y_pred_proba = model.predict_proba(X_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        print("auc="+str(auc))
        plt.legend(loc=4)
        plt.show()

    def s_kfold(self,model):
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        scores = cross_val_score(model, X_train, y_train, cv=skfold,scoring='accuracy')
        print('\nCross-Validation Accuracy Scores', scores)

        scores = pd.Series(scores)
        print('\nThe minimum Cross-Validation Accuracy is  %.4f ' % scores.min())
        print('\nThe mean Cross-Validation Accuracy is  %.4f ' % scores.mean())
        print('\nThe maximum Cross-Validation Accuracy is  %.4f ' % scores.max())

    def kfold(self,model):
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)

        scores = cross_val_score(model, X_train, y_train, cv=kfold,scoring='accuracy')
        print('\nCross-Validation Accuracy Scores', scores)

        scores = pd.Series(scores)
        print('\nThe minimum Cross-Validation Accuracy is  %.4f ' % scores.min())
        print('\nThe mean Cross-Validation Accuracy is  %.4f ' % scores.mean())
        print('\nThe maximum Cross-Validation Accuracy is  %.4f ' % scores.max())

    def calc_metrics(self,y_pred):
        print("\nF1 Score: %.4f " % metrics.f1_score(y_test, y_pred))
        print("\nAccuracy: %.4f " % metrics.accuracy_score(y_test, y_pred))
        print("\nPrecision: %.4f " % metrics.precision_score(y_test, y_pred))
        print("\nRecall: %.4f " % metrics.recall_score(y_test, y_pred))

def logistic(X_train,y_train,X_test,y_test):
    # instantiate the model (using the default parameters)
    logistic_regressor = LogisticRegression()

    # fit the model with data
    logistic_regressor = logistic_regressor.fit(X_train,y_train)

    # predict
    y_pred = logistic_regressor.predict(X_test)

    helper = helper()
    helper.s_kfold(logistic_regressor)
    helper.kfold(logistic_regressor)
    helper.roc_plot(logistic_regressor)
    helper.calc_metrics(y_pred)

def boosting(X_train,y_train,X_test,y_test):
    # instantiate the model (using the default parameters)
    xgboost_classifier = XGBClassifier()

    # fit the model with data
    xgboost_classifier = xgboost_classifier.fit(X_train,y_train)

    # predict
    y_pred = xgboost_classifier.predict(X_test)

    helper = helper()
    helper.s_kfold(xgboost_classifier)
    helper.kfold(xgboost_classifier)
    helper.roc_plot(xgboost_classifier)
    helper.calc_metrics(y_pred)

def perceptron(X_train,y_train,X_test,y_test):
    # create mutli-layer perceptron classifier
    perceptron_classifier = MLPClassifier()

    # train
    perceptron_classifier = perceptron_classifier.fit(X, y)

    # make predictions
    y_pred = perceptron_classifier.predict(X_test)

    helper = helper()
    helper.s_kfold(perceptron_classifier)
    helper.kfold(perceptron_classifier)
    helper.roc_plot(perceptron_classifier)
    helper.calc_metrics(y_pred)

def svm(X_train,y_train,X_test,y_test):
    #Create a svm Classifier
    support_vector_classifier = svm.SVC(kernel='linear', probability=True)

    #Train the model using the training sets
    support_vector_classifier = support_vector_classifier.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = support_vector_classifier.predict(X_test)

    helper = helper()
    helper.s_kfold(support_vector_classifier)
    helper.kfold(support_vector_classifier)
    helper.roc_plot(support_vector_classifier)
    helper.calc_metrics(y_pred)

def svm(X_train,y_train,X_test,y_test):
    # Create Decision Tree classifer object
    decision_tree_classifer = DecisionTreeClassifier(max_depth=5)

    # Train Decision Tree Classifer
    decision_tree_classifer = decision_tree_classifer.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = decision_tree_classifer.predict(X_test)

    helper = helper()
    helper.s_kfold(decision_tree_classifer)
    helper.kfold(decision_tree_classifer)
    helper.roc_plot(decision_tree_classifer)
    helper.calc_metrics(y_pred)

def svm(X_train,y_train,X_test,y_test):
    # create classifier object 
    random_forest_classifier = RandomForestClassifier() 
    
    # fit the classifier with x and y data 
    random_forest_classifier = random_forest_classifier.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = random_forest_classifier.predict(X_test)

    helper = helper()
    helper.s_kfold(random_forest_classifier)
    helper.kfold(random_forest_classifier)
    helper.roc_plot(random_forest_classifier)
    helper.calc_metrics(y_pred)