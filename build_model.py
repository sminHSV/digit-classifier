import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize values
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential()

# pad the 28x28 input to 32x32
model.add(tf.keras.layers.ZeroPadding2D(2, input_shape=(28,28,1)))
# randomly rotate, scale, and translate the image to reduce overfitting
model.add(tf.keras.layers.RandomRotation(0.05, fill_mode='constant'))
model.add(tf.keras.layers.RandomZoom((-0.1, 0.0), (-0.1, 0.0), fill_mode='constant'))
model.add(tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode='constant'))
# 5x5 convolutional layer
model.add(tf.keras.layers.Conv2D(6, (5, 5), activation='relu'))
# 2x2 pooling layer
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2))
# 5x5 convolutional layer
model.add(tf.keras.layers.Conv2D(16, (5, 5), activation='relu'))
# 2x2 pooling layer
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2))
model.add(tf.keras.layers.Flatten())
# fully connected layer (120 units)
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
# fully connected layer (84 units)
model.add(tf.keras.layers.Dense(84, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
# output layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=64)

model.save('saved_models/model.keras')

saved_model = tf.keras.models.load_model('saved_models/model.keras')
saved_model.evaluate(x_test, y_test)