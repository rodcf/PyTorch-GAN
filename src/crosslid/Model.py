import numpy as np
import onnx
import onnxruntime as ort
import yaml

from onnx2keras import onnx_to_keras

from src.crosslid.utilities import scale_value

with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)

def get_ort_session(onnx_path:str):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    return ort_session

def get_onnx_model(onnx_path:str):
    onnx_model = onnx.load(onnx_path)# Call the converter (input will be equal to the input_names parameter that you defined during exporting)
    return onnx_to_keras(onnx_model, ['input'])

def get_fake_images(onnx_path:str, model:str, size:int=20000):
    print('Generating fake images using trained model:', model.upper())
    if model != 'cgan':
        model = get_onnx_model(onnx_path)
        noise = np.random.normal(size=(size, params['latent_dim']))
        images = model.predict(noise)
    else:
        ort_session = get_ort_session(onnx_path)
        noise = np.random.normal(size=(size, params['latent_dim'])).astype(np.float32)
        labels = np.random.randint(0, 10, size)
        images = ort_session.run(None, {'input':noise, 'labels': labels})
    images = scale_value(images, [0, 255.0])
    return images