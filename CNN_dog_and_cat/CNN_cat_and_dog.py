import warnings

import matplotlib.pyplot as plt
import tensorflow as tf

warnings.filterwarnings('ignore')

from keras import layers
from keras.utils import image_dataset_from_directory

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import matplotlib.image as mpimg

import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

_URL = 'https://www.kaggle.com/api/v1/datasets/download/subho117/cat-and-dog-classification-using-cnn?dataset_version_number=1'
zip_dir = tf.keras.utils.get_file('archive.zip', origin=_URL, extract=True)

zip_dir_base = os.path.dirname(zip_dir)

base_dir = os.path.join(os.path.dirname(zip_dir), 'dog-vs-cat')

fig = plt.gcf()
fig.set_size_inches(16, 16)
cat_dir = os.path.join(base_dir, 'cat')
dog_dir = os.path.join(base_dir, 'dog')
cat_names = os.listdir(cat_dir)
dog_names = os.listdir(dog_dir)

pic_index = 210

cat_images = [os.path.join(cat_dir, fname)
              for fname in cat_names[pic_index - 8:pic_index]]
dog_images = [os.path.join(dog_dir, fname)
              for fname in dog_names[pic_index - 8:pic_index]]

for i, img_path in enumerate(cat_images + dog_images):
    sp = plt.subplot(4, 4, i + 1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

train_datagen = image_dataset_from_directory(base_dir,
                                             image_size=(200, 200),
                                             subset='training',
                                             seed=1,
                                             validation_split=0.1,
                                             batch_size=32)
test_datagen = image_dataset_from_directory(base_dir,
                                            image_size=(200, 200),
                                            subset='validation',
                                            seed=1,
                                            validation_split=0.1,
                                            batch_size=32)

model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(train_datagen, epochs=10, validation_data=test_datagen)
model.save('cnn_cat_and_dog')

#
# history_df = pd.DataFrame(history.history)
# history_df.loc[:, ['loss', 'val_loss']].plot()
# history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
# plt.show()
#
# !wget https://i.pinimg.com/originals/9c/02/d2/9c02d21c245a443f4a2e2c182f268808.jpg -O /content/image.png
#
# !wget https://i.pinimg.com/736x/2e/03/6e/2e036e60cdd1ff8a036f3e76184312ec.jpg -O /content/image1.png
#
# from keras.preprocessing import image
#
# #Input image
# test_image = image.load_img('/content/image.png',target_size=(200,200))
#
# #For show image
# plt.imshow(test_image)
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image,axis=0)
#
# # Result array
# result = model.predict(test_image)
#
# #Mapping result array with the main name list
# i=0
# if(result>=0.5):
#   print("Dog")
# else:
#   print("Cat")
#
# from keras.preprocessing import image
#
# #Input image
# test_image = image.load_img('/content/image1.png',target_size=(200,200))
#
# #For show image
# plt.imshow(test_image)
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image,axis=0)
#
# # Result array
# result = model.predict(test_image)
#
# #Mapping result array with the main name list
# i=0
# if(result>=0.5):
#   print("Dog")
# else:
#   print("Cat")
