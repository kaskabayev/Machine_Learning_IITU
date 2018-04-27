# Predicting using model
from keras.models import model_from_json
from keras.preprocessing import image

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Model loaded")

loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Predicting
import numpy as np
from keras.preprocessing import image

for i in range(1, 8):
    test_image = image.load_img('dataset/single_prediction/cat_or_dog_' + str(i) + '.jpg', target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict(test_image)

    if(result[0][0] == 1):
        prediction = 'dog'
    else:
        prediction = 'cat'

    print(i, prediction)