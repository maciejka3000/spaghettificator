import yaml
import os
import glob
from dataclasses import dataclass
import moonrakerpy as mpy
import cv2 as cv
from io import BytesIO
from urllib import request
import numpy as np

@dataclass
class Spaghettificator:
    # hidden vars
    __path_main: str
    __path_code: str
    __path_sets: str
    __path_modfolder: str
    __path_model: str
    
    # Yaml data
    __url_3d: str
    __url_ph: str
    __execution_env: str
    __model_settings: dict
    
    __printer: mpy
    
    # public vars
    
    
    # init function
    def __init__(self):
        self.__path_code = os.path.dirname(__file__)
        self.__path_main = os.path.dirname(self.__path_code)
        self.__path_modfolder = os.path.join(self.__path_main, "model")
        self.__path_sets = os.path.join(self.__path_main, "settings.yaml")
        
        # YAML reading
        with open(self.__path_sets, 'r') as yamlfile:
            yamldata = yaml.safe_load(yamlfile)
            
        # YAML processing
        
        #1. load data stored in YAML
        self.__load_yaml(yamldata)
        #2. load execution enviroment
        self.__load_env(yamldata)

        
        
        # Connecting to API
        self.__printer = mpy.MoonrakerPrinter(self.__url_3d)
        
    # PRIVATE METHODS
    
    def __load_yaml(self, yamldata: dict) -> None:
        
        # Check config
        if type(yamldata['photos_url']) != str:
            raise ValueError("settings.yaml/photos_url: must be string")
        
        ex_env = yamldata['execution_enviroment']
        exe_check_str = ex_env == "openvino" or ex_env == "onnx"
        exe_check_typ = type(ex_env) == str
        
        # execution_enviroment test
        if not(exe_check_str and exe_check_typ):
            raise ValueError("settings.yaml/execution_enviroment: Invalid setting. Must be 'openvivo' or 'onnx' string.")
        
        # model_name test
        try:
            mod_path = os.path.join(self.__path_modfolder, yamldata['model_name'])
        except:
            raise ValueError("settings.yaml/model_name: Invalid setting. Must be string.")
        
        path_exists = os.path.exists(mod_path)
        if not(path_exists):
            raise ValueError("settings.yaml/model_name: Invalid setting. Model can not be found.")
        
        #classification_settings
        cls_settings = yamldata['classification_settings']
        if cls_settings['cls_detecting_gap'] <= 0:
            raise ValueError("settings.yaml/classification_settings/cls_detecting_gap: Value must be greater than 0")
            
        if cls_settings['cls_sensitivity'] > 1 or cls_settings['cls_sensitivity'] <= 0:
            raise ValueError("settings.yaml/classification_settings/cls_sensitivity: Invalid value. Value must be between 0 and 1.")
        
        e_gap = cls_settings['cls_ensuring_gap']
        if not(isinstance(e_gap, (int, float))) or e_gap <= 0:
            raise ValueError("settings.yaml/classification_settings/cls_ensuring_gap: Invalid value. Value must be bigger than 0")
        
        encomp = cls_settings['cls_ensuring_ncomp']
        if not(isinstance(encomp, int)) or encomp <= 0:
            raise ValueError("settings.yaml/classification_settings/cls_ensuring_ncomp: Invalid value. Value must be int, greater than 0")
        
        ensens = cls_settings['cls_ensuring_sensitivity']
        if not(isinstance(ensens, (float, int))) or ensens >= 1 or ensens <= 0:
            raise ValueError("settings.yaml/classification_settings/cls_ensuring_sensitivity: Invalid value. Value must be between 0 and 1")
        
        detgcode = cls_settings['spaghetti_detected_gcode']
        if not(isinstance(detgcode, (str))):
            raise ValueError("settings.yaml/classification_settings/spaghetti_detected_gcode: Invalid value. Value must be string")
        
        
        # after check.
        # Get URLs
        
        if not yamldata['printer_url']:
            self.__url_3d = "localhost"
        else:
            self.__url_3d = yamldata['printer_url']
        self.__url_3d = ''.join(["http://", self.__url_3d])
        self.__url_ph = ''.join([self.__url_3d, yamldata['photos_url']])
        
        # Get model path
        self.__path_model = os.path.join(self.__path_modfolder, yamldata['model_name'])
        
        # Get execution env
        self.__execution_env = ex_env
        
        # Get model settings
        self.__model_settings = cls_settings
        
    def __load_env(self, yamldata: dict) -> None:
        envname = self.__execution_env
        
        if envname == "openvino":
            import openvino_frame as ovf
            # load model and assign class to that model
            self.__model = ovf.Env_Vivo(self.__path_model)
            self.__modelClassificationMethod = self.__model.Classify
        
        elif envname == "onnx":
            import onnx_frame as onf
            self.__model = onf.Env_onnx(self.__path_model)
        
        else:
            raise(ValueError, "settings.yaml/execution_enviroment: Invalid setting.")
        

        
        
    # PUBLIC METHODS
    def Get_printing_status(self) -> bool:
        """Get_printing_status: Get status of 3d printer. Function returns 'True' during
        printing. Otherwise, function returns 'False'

        Returns:
            bool: Printing status
        """
        status = self.__printer.get_gcode(1)
        statusmap = status[0][0:2]
        if statusmap == "//" or statusmap == "!!":
            return False
        else:
            return True
    
    def Get_image(self) -> np.ndarray:
        """Get_image: Import image from 3d printer server, and return it as a np.ndarray.

        Returns:
            np.ndarray: imported image from the server.
        """
        response = request.urlopen(self.__url_ph)
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv.imdecode(arr, -1)
        return img
    
    def Classify(self, img: np.ndarray = None) -> dict:
        """Classify: Generate predictions of image, using determined execution enviroment.

        Args:
            img (np.ndarray, optional): Image to get classification. If image will not be
            determined, then function 'get_image' will be used to get one. Defaults to None.

        Returns:
            dict: classifications and their keys.
        """
        if type(img) == type(None):
            img = self.Get_image()
        
        return self.__modelClassificationMethod(img)
    
    
        
        


        
        
    
        
        
        
        
        
x = Spaghettificator()
img = x.Get_image()
data = x.Classify()
print(data)


