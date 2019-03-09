"""This is the example codes for tensorflow with tf.keras
"""
import tensorflow as tf
from tensorflow.keras import layers

USE_KERAS_API = False
num_classes = 10

tf.enable_eager_execution()


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() 
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

# Get original data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data() 

#Others
# def load_and_preprocess_from_path_label(path, label):
#     return load_and_preprocess_image(path), label
# image_label_ds = ds.map(load_and_preprocess_from_path_label)
dataset = tf.data.Dataset.from_tensor_slices( (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32), 
tf.cast(mnist_labels,tf.int64))) 
dataset = dataset.shuffle(1000).batch(32) 


# Get original data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data() 

#Others
# def load_and_preprocess_from_path_label(path, label):
#     return load_and_preprocess_image(path), label
# image_label_ds = ds.map(load_and_preprocess_from_path_label)
dataset = tf.data.Dataset.from_tensor_slices( (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32), 
tf.cast(mnist_labels,tf.int64))) 
dataset = dataset.shuffle(1000).batch(32) 


'''
base_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)
base_feature = base_model.output
feature = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(base_feature)
prediction = tf.keras.layers.Dense(num_classes, name='predictions')(feature)
model = tf.keras.Model(inputs=base_model.input, outputs=prediction)
'''
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))

if USE_KERAS_API:
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    model.fit(x_train, y_train, 
              epochs=10, batch_size=256,
              validation_data=(x_test, y_test))
else:
    def compute_loss(logits, labels, model):
        loss_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
        #loss_reg = tf.add_n(model.losses)
        total_loss = loss_cross_entropy 
        # Scale loss by global batch size.
        return total_loss

    for (batch, (images, labels)) in enumerate(iter(dataset)):
        with tf.GradientTape() as tape: 
            logits = model(images, training=True)
            _loss = compute_loss(logits, labels, model) 
            print(_loss)
        grads = tape.gradient(_loss, model.trainable_variables)   
        tf.train.AdamOptimizer().apply_gradients(zip(grads, model.trainable_variables), 
                                            global_step=tf.train.get_or_create_global_step()) 
        break