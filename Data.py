def encoder(dataset):
    # Replacing categorical values with numerical values
    label_encoder = LabelEncoder()
    all_columns = list(dataset.columns)
    for x in all_columns:
        if type(dataset[x][0]) == str:
            try:
                dataset[x] = label_encoder.fit_transform(dataset[x])
            except:
                continue
    return dataset

def outlier(dataset):
    all_columns = list(dataset.iloc[:,:20].columns)
    for x in all_columns:
        try:
            dataset[x] = np.where(dataset[x] > dataset[x].quantile(0.975), dataset[x].quantile(0.50), dataset[x])
            dataset[x] = np.where(dataset[x] < dataset[x].quantile(0.025), dataset[x].quantile(0.50), dataset[x])
        except TypeError:
            continue
    return dataset

def scaler(dataset):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(dataset.iloc[:,:20])
    dataset_additional_fullish = pd.DataFrame(scaled_df,columns = all_columns)
    dataset_fullish['y'] = dataset['y']
    dataset = dataset_fullish
    return dataset

def features(dataset):
    X = dataset[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan','contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays','previous', 'poutcome', 'emp.var.rate', 'cons.price.idx','cons.conf.idx', 'euribor3m', 'nr.employed']]
    y = dataset['y']
    return dataset

def tsne(X):
    tsne = TSNE(n_components=3)
    X_tsne = tsne.fit_transform(X)
    X_tsne = pd.DataFrame(X_tsne)
    return X_tsne

def autoencode(X):
    # Choose size of the encoded representations (reduce our initial features to this number)
    encoding_dim = 15
    # Define input layer
    input_data = Input(shape=(X.shape[1],))
    # Define encoding layer
    encoded = Dense(encoding_dim, activation='elu')(input_data)
    # Define decoding layer
    decoded = Dense(X.shape[1], activation='sigmoid')(encoded)
    # Create the autoencoder model
    autoencoder = Model(input_data, decoded)
    #Compile the autoencoder model
    autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
    #Fit to set and save to hist_auto for plotting purposes
    hist_auto = autoencoder.fit(X, X,epochs=500,batch_size=256,shuffle=True)
    # Create a separate model (encoder) in order to make encodings (first part of the autoencoder model)
    encoder = Model(input_data, encoded)

    # Create a placeholder for an encoded input
    encoded_input = Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    # Encode and decode our test set (compare them visually just to get a first insight of the autoencoder's performance)
    encoded_X = encoder.predict(X)
    decoded_output = decoder.predict(encoded_X)

    #Encode data set from above using the encoder
    encoded_X = encoder.predict(X)

    model = Sequential()
    model.add(Dense(16, input_dim=encoded_X.shape[1],kernel_initializer='normal',activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer='adam')

    history = model.fit(encoded_X, y, validation_split=0.2, epochs=100, batch_size=64)
    #Predict on set
    predictions_prob = model.predict(encoded_X)
    predictions_prob = predictions_prob[:,0]

    predictions = np.where(predictions_prob > 0.5, 1, 0) 
    #Turn probability to 0-1 binary output

    #Print accuracy
    acc = metrics.accuracy_score(y, predictions)
    print('Overall accuracy of Neural Network model:', acc)

    decoded_output = pd.DataFrame(decoded_output)
    return decoded_output

def principal(X):
    pca = PCA()
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca)
    return X_pca
