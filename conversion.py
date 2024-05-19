import tf2onnx
from keras.models import load_model
import tensorflow as tf

keras_model = load_model('./model/image_classifier_model1.h5')

functional_model = tf.keras.Model(inputs=keras_model.input, outputs=keras_model.output)
# Convert the model to ONNX format
onnx_model, _ = tf2onnx.convert.from_keras(functional_model)
print(onnx_model)

# import onnx
# from onnx2pytorch import ConvertModel

# # Convert ONNX model to PyTorch
# pytorch_model = ConvertModel(onnx_model)
# print(pytorch_model)
