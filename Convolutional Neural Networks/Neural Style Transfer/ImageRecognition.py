# Neural Style Transfer

# Load packages
import numpy as np
from PIL import Image

from keras import backend
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

height = 512
width = 512

combination_image = backend.placeholder((1, height, width, 3))

# fetching a pretrained vgg16 CNN using the ImageNet database
model = VGG16(weights = 'imagenet', include_top=True)
# let's check every layer in the CNN
layers = dict([(layer.name, layer.output) for layer in model.layers])
# print the different layer characteristics
layers
# How many parameters are in the model?
model.count_params()

# load an image and classify it
image_path = 'Elephant.jpg'
image = Image.open(image_path)
image = image.resize((224, 224))
image.show()

# Convert it into an array
x = np.asarray(image, dtype='float32')
# Convert it into a list of arrays
x = np.expand_dims(x, axis=0)
# Pre-process the input to match the training data
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])