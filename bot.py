import tensorflow as tf

# loads minst dataset 
mnist = tf.keras.datasets.mnist

# converts from integers to floating point numbers 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# create a keras sequential model by stacking layers
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

#  the model returns a vector of logits or log-odds scores, one for each class
predictions = model(x_train[:1]).numpy()
predictions

# converts these logits to probabilities for each class
tf.nn.softmax(predictions).numpy()

# takes a vector of logits and a truee index and returns a scalar loss for each example
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class
loss_fn(y_train[:1], predictions).numpy()\
    
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=7000)

# checks the models performance 
model.evaluate(x_test,  y_test, verbose=2)


#probability_model = tf.keras.Sequential([
#  model,
#  tf.keras.layers.Softmax()
#])

#probability_model(x_test[:5])

