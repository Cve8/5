import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


#### TRAIN ####
loaded = np.load("raw_data.npz")
trainIngredients = loaded["trainIngredients"]
trainCuisine = loaded["trainCuisine"]
allIngredients = loaded["allIngredients"]

boolTrainIngredients = np.zeros((len(trainIngredients), len(allIngredients)))
for i in range(len(trainIngredients)):
    recipeIngredients = trainIngredients[i]
    for j in range(len(allIngredients)):
        if allIngredients[j] in recipeIngredients:
            boolTrainIngredients[i][j] = 1

vectorizer = CountVectorizer(input = "content")
boolTrainCuisine = vectorizer.fit_transform(trainCuisine).toarray()
boolTrainCuisine = boolTrainCuisine.argmax(1)

#### TEST ####
test = np.load("test.npz")
boolTestIngredients = test["boolTestIngredients"]
boolTestCuisine = test["boolTestCuisine"]

#### NEURALNET ####
neural_net = Sequential()
neural_net.add(Dense(100, activation='relu', input_shape = (boolTrainIngredients.shape[1],)))
neural_net.add(Dropout(0.1))
neural_net.add(Dense(100, activation='relu'))
neural_net.add(Dense(20, activation='softmax'))
neural_net.summary()

neural_net.compile(optimizer="Adamax", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
history = neural_net.fit(boolTrainIngredients, boolTrainCuisine, verbose=1, validation_data=(boolTestIngredients, boolTestCuisine), epochs=10)
yTest = neural_net.predict(boolTestIngredients)
predictLabels = np.argmax(yTest, axis = 1)
print("#### NEURALNET ####")
for i in range(len(predictLabels)):
    for j in range(len(boolTrainCuisine)):
        if boolTrainCuisine[j] == predictLabels[i]:
            cuisineLabel = trainCuisine[j]
    print(i, "predicted label:", cuisineLabel)
loss, accuracy = neural_net.evaluate(boolTestIngredients, boolTestCuisine, verbose=0)
print("NN accuracy: {}%".format(accuracy*100))


#### RANDOMFOREST ####
forest = RandomForestClassifier(n_estimators=100, max_features='auto', class_weight='balanced')
forest.fit(boolTrainIngredients, boolTrainCuisine)
yTest = forest.predict(boolTestIngredients)
print("#### RANDOMFOREST ####")
for i in range(len(yTest)):
    for j in range(len(boolTrainCuisine)):
        if boolTrainCuisine[j] == yTest[i]:
            cuisineLabel = trainCuisine[j]
    print(i, "predicted label", cuisineLabel)
accuracy = forest.score(boolTestIngredients, boolTestCuisine)
print("RF accuracy: {}%".format(accuracy*100))
