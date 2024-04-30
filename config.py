import json
from typing import Dict

class Config:
    def __init__(self, config_path: str = 'config.json') -> None:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.__dict__.update(data)
    
    @property
    def dict(self) -> Dict:
        return self.__dict__
    
    def __str__(self) -> str:
        return str(self.__dict__)
    
    def __repr__(self) -> str:
        return str(self.__dict__)
    

if __name__ == "__main__":
    config = Config('config.json')
    print(config.dict)
        