import numpy
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout

# The competition datafiles are in the directory ../input
# Read competition data files:
train_file = "train.csv"
test_file = "test.csv"
output_file = "submission.csv"

# Image size 
img_rows, img_cols = 28, 28
# Number of pixels
num_pixels = img_rows * img_cols
# Size of mini batch
batch_size = 128
# Epochs amount
nb_epoch = 20

# Loading data
mnist_dataset = numpy.loadtxt(train_file, skiprows=1, dtype='int', delimiter=",")

# Shuffle the records
numpy.random.shuffle(mnist_dataset)

# Separating data and labels
x = mnist_dataset[:, 1:]
y = mnist_dataset[:, 0]

# Normalizing data
x = x.astype("float32")
x /= 255.0

# Converting the labels to the catecories
y = np_utils.to_categorical(y)

nb_classes = y.shape[1]

# Creating a sequential model
model = Sequential()
# Adding the input layer
model.add(Dense(800, input_dim=num_pixels, init="normal", activation="relu"))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(nb_classes, init="normal", activation="softmax"))

# Compiling the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Printing the model summary
print(model.summary())

# Fitting the model
model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

# Loading test data          
x_test = numpy.loadtxt(test_file, skiprows=1, delimiter=",")
x_test /= 255.0

# Making predictions
predictions = model.predict(x_test)

# Converting the categories to the labels
predictions = np_utils.categorical_probas_to_classes(predictions)

#Writing data to the output
out = numpy.column_stack((range(1, predictions.shape[0]+1), predictions))
numpy.savetxt(output_file, out, header="ImageId,Label", comments="", fmt="%d,%d")
