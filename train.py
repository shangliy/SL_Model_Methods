from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

CLASS_NUM = 6
img_width, img_height = 299, 299
train_data_dir = '/media/shangliy/Storage/Source/realreal_triplet'
nb_train_samples = 100
nb_epoch = 10

# The input data
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=1)

# creat the base pre-trained model
base_model = InceptionV3(weights='imagenet',include_top=False)

# add a global spatial averange pooling layers
x = base_model.output

#model = Model(input=base_model.input, output=base_model.output)

#First: train only the top layers(which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

#x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
#x = Dense(1024,activation='relu')(x)
# add a logistic layer, suppose we have 64 classes
#predictions = Dense(CLASS_NUM, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=base_model.output)

#First: train only the top layers(which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

#train the model on the new data

feature = model.predict_generator(train_generator,
                        val_samples = 10)
print(feature.shape)
