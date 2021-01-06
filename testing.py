import seaborn as sbn
from train import m, val
import numpy as np

from validate_daata import viz_my_data

predictions = m.predict(val['images'])
sbn.distplot(predictions[:, 0]);

predicted_label = np.argmax(predictions, axis=-1)
print('accuracy:', np.mean(predicted_label == val['labels']))
viz_my_data(num=(6, 6), predictions=predictions[:, 1], **val);

# save the model
m.save("model.h5")

# If you want to make sure that this model can be used on different operating systems and different
# versions of keras or tensorflow, this is the better way to save. For this project the simpler
# method above should work fine.

# json_filename = 'model.json'
# h5_filename   = 'weights.h5'
# # create a json with the model architecture
# model_json = m.to_json()
# # save the json to disk
# with open(json_filename, 'w') as f:
#     f.write(model_json)
# # save the model's weights:
# m.save_weights(h5_filename)
# print(" ".join(["Model saved to", json_filename, h5_filename]))

# Loading the model

from tensorflow.keras.models import load_model

loaded_model = load_model("model.h5")

# If you use the more robust method of saving above, this is how you load the model.

# with open(json_filename, 'r') as j:
#     loaded_json = j.read()

# # load the model architecture:
# loaded_model = keras.models.model_from_json(loaded_json)
# #load the weights:
# loaded_model.load_weights(h5_filename)
# print(" ".join(["Model loaded from", json_filename, h5_filename]))


# If you use the more robust method of saving above, this is how you load the model.

# with open(json_filename, 'r') as j:
#     loaded_json = j.read()

# # load the model architecture:
# loaded_model = keras.models.model_from_json(loaded_json)
# #load the weights:
# loaded_model.load_weights(h5_filename)
# print(" ".join(["Model loaded from", json_filename, h5_filename]))
# code copied from the training evaluation:
l_predictions = loaded_model.predict(val['images'])
sbn.distplot(l_predictions[:, 0]);

l_predicted_label = np.argmax(l_predictions, axis=-1)
print('accuracy:', np.mean(l_predicted_label == val['labels']))
