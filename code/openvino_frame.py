import openvino as ov
import cv2 as cv
import numpy as np
import ast
from dataclasses import dataclass



# Dataclass used to get classification results using openVIVO toolkit. 
@dataclass
class Env_Vivo:

    # private vars
    __compiled_model: ov.CompiledModel
    __dtype: np.dtypes
    __classes: tuple
    __size_img: tuple
    
    # private methods
    def __init__(self, model_path: str, classes:tuple = ("spaghetti", "nospaghetti"))-> None:
        # Load and compile model
        model = ov.convert_model(model_path)
        self.__compiled_model = ov.compile_model(model)
        
        # Get proper type of input images
        self.__dtype = self.__compiled_model.inputs[0].element_type.to_dtype()
        
        # Find size of input images
        shapesize = ast.literal_eval(self.__compiled_model.inputs[0].shape.to_string())
        self.__size_img = tuple(shapesize[2: 4])
        
        # Write classification types to class
        self.__classes = classes
        
    # Preprocess image to get a photo which is properly formatted to be classified by model
    def __imPreprocess(self, img) -> np.ndarray:
        img = cv.resize(img, self.__size_img)
        img = img / 255.0
        img = img.astype(self.__dtype)
        img = np.expand_dims(img, axis=0)
        return np.transpose(img, (0, 3, 1, 2))
    
    ## PUBLIC METHODS
    """
    Classify: Classify image using openVIVO toolkit.
    
    Parameters:
        im_path (str): path to classified image.
        
    Returns:
        di (dict):  Keys of this dict are the classification classes.
                    The values of keys are the confidence score of possible classes. 
    """
    def Classify(self, img: np.ndarray) -> dict:
        # Preprocess an image
        img = self.__imPreprocess(img)
        
        # Classify image, then get scores
        result = self.__compiled_model(img)
        scores = result[0][0].tolist()
        
        # zip classes and scores together, return it as dict
        di = dict(zip(self.__classes, scores))
        
        return di