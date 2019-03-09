from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten,Reshape
from keras.models import Model
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# first, define the vision modules
digit_input = Input(shape=(27, 27,1))
x = Convolution2D(64, 3, 3)(digit_input)
x = Convolution2D(64, 3, 3)(x)
x = MaxPooling2D((2, 2))(x)
out = Flatten()(x)

print Reshape

vision_model = Model(digit_input, out)

# then define the tell-digits-apart model
digit_a = Input(shape=(27, 27,1))
digit_b = Input(shape=(27, 27,1))

# the vision model will be shared, weights and all
out_a = vision_model(digit_a)
reshape_a = Reshape(target_shape=(3872 *2,))(out_a)
out_b = vision_model(digit_b)
reshape_b = Reshape(target_shape=(3872 *2,))(out_b)

concatenated = merge([reshape_a, reshape_b], mode='concat')

out = Dense(1, activation='sigmoid')(concatenated)
print concatenated