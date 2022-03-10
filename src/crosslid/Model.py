import numpy as np
import onnx
from onnx2keras import onnx_to_keras

from src.crosslid.utilities import scale_value

def get_onnx_model(onnx_path:str):
    onnx_model = onnx.load(onnx_path)# Call the converter (input will be equal to the input_names parameter that you defined during exporting)
    return onnx_to_keras(onnx_model, ['input'])

def get_fake_images(onnx_path:str, size:int=20000):
    model = get_onnx_model(onnx_path)
    print('Generating fake images using trained model')
    noise = np.random.normal(size=(size, 100))
    images = model.predict(noise)
    images = scale_value(images, [0, 255.0])
    return images