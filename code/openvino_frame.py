import openvino as ov
import cv2 as cv
import numpy as np
from dataclasses import dataclass


@dataclass
class Env_Vivo:

    # private vars
    __compiled_model: ov.CompiledModel
    __dtype: np.dtypes
    __classes: tuple
    # public vars
    
    def __init__(self, model_path: str, classes:tuple = ("spaghetti", "nospaghetti"))-> None:
        model = ov.convert_model(model_path)
        self.__compiled_model = ov.compile_model(model)
        self.__dtype = self.__compiled_model.inputs[0].element_type.to_dtype()
        self.__classes = classes
        
    def __imPreprocess(self, im_path) -> np.ndarray:
        img = cv.imread(im_path)
        img = cv.resize(img, (256, 256))
        img = img / 255.0
        img = img.astype(self.__dtype)
        img = np.expand_dims(img, axis=0)
        return np.transpose(img, (0, 3, 1, 2))
    
    def Classify(self, im_path: str):
        img = self.__imPreprocess(im_path)
        result = self.__compiled_model(img)
        scores = result[0][0].tolist()
        di = dict(zip(self.__classes, scores))
        
        return di

        




path = "/home/maciejka/Documents/projects/spaghettificator/spaghettificator/model/model_raw.onnx"

path_img = "/home/maciejka/Documents/projects/spaghettificator/training-models/3.jpg"

env = Env_Vivo(path)
scr = env.Classify(path_img)
print(scr)


