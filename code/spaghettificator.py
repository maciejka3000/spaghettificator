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
    
    __yaml_3d_url: str
    __yaml_ph_url: str
    
    __yaml_execution_enviroment: str
    
    __model_settings: dict
    __printer: mpy
    
    # public vars
    
    # init function
    def __init__(self):
        self.__path_code = os.path.dirname(__file__)
        self.__path_main = os.path.dirname(self.__path_code)
        self.__path_sets = os.path.join(self.__path_main, "settings.yaml")
        
        # YAML reading
        with open(self.__path_sets, 'r') as yamlfile:
            yamldata = yaml.safe_load(yamlfile)
            
        # YAML processing
        
        #1. get urls
        self.__loadurls(yamldata)
        #2. load execution enviroment
        self.__load_env(yamldata)
        #3. get classification settings
        self.__loadsettings(yamldata)
        
        
        # Connecting to API
        self.__printer = mpy.MoonrakerPrinter(self.__yaml_3d_url)
        
    # PRIVATE METHODS
    
    def __load_env(self, yamldata: dict) -> None:
        self.__yaml_execution_enviroment = yamldata['execution_enviroment']
        
        
        
    def __loadsettings(self, yamldata: dict) -> None:
        self.__model_settings = yamldata['classification_settings']
        

        
    def __loadurls(self, yamldata: dict) -> None:
        if not yamldata['printer_url']:
            self.__yaml_3d_url = "localhost"
        else:
            self.__yaml_3d_url = yamldata['printer_url']
        self.__yaml_3d_url = ''.join(["http://", self.__yaml_3d_url])
        self.__yaml_ph_url = ''.join([self.__yaml_3d_url, yamldata['photos_url']])
        
    # PUBLIC METHODS
    def get_printing_status(self) -> bool:
        status = self.__printer.get_gcode(1)
        statusmap = status[0][0:2]
        if statusmap == "//" or statusmap == "!!":
            return False
        else:
            return True
    
    def get_image(self) -> np.ndarray:
        response = request.urlopen(self.__yaml_ph_url)
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv.imdecode(arr, -1)
        return img
    
        
        


        
        
    
        
        
        
        
        
x = Spaghettificator()
img = x.get_image()

print(type(img))

