import yaml
import os
import glob
from ultralytics import YOLO
from dataclasses import dataclass
import moonrakerpy as mpy

@dataclass
class Spaghettificator:
    # hidden vars
    __path_main: str
    __path_code: str
    __path_sets: str
    
    __yaml_3d_url: str
    __yaml_ph_url: str
    
    __model_settings: dict
    __model: YOLO
    
    __printer: mpy
    
    # public vars
    
    # init function
    def __init__(self):
        self.__path_code = os.getcwd()
        self.__path_main = os.path.dirname(self.__path_code)
        self.__path_sets = os.path.join(self.__path_main, "settings.yaml")
        
        # YAML reading
        with open(self.__path_sets, 'r') as yamlfile:
            yamldata = yaml.safe_load(yamlfile)
            
        # YAML processing
        
        #1. get urls
        self.__loadurls(yamldata)
        #2. get model
        self.__loadmodel(yamldata)
        #3. get classification settings
        self.__loadsettings(yamldata)
        
        # Connecting to API
        __printer = mpy.MoonrakerPrinter(self.__yaml_3d_url)
        
    # PRIVATE METHODS
        
    def __loadsettings(self, yamldata: dict):
        self.__model_settings = yamldata['classification_settings']
    
    def __loadmodel(self, yamldata: dict):
        model_name = yamldata['model_name']
        model_path = os.path.join(self.__path_main, "model", model_name)
        self.__model = YOLO(model_path)
        
    def __loadurls(self, yamldata: dict):
        if not yamldata['printer_url']:
            self.__yaml_3d_url = "localhost"
        else:
            self.__yaml_3d_url = yamldata['printer_url']
        self.__yaml_3d_url = ''.join(["http://", self.__yaml_3d_url])
        self.__yaml_ph_url = ''.join([self.__yaml_3d_url, yamldata['photos_url']])
        
    # PUBLIC METHODS
    
    
        
        
        
        
        
x = Spaghettificator()
