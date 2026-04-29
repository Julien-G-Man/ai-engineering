"""
EXERCISE 3 — JSON Config Loader 

Write load_config(path: str) -> dict that reads a JSON file and returns its contents, returning {} if the file 
doesn't exist 
Write save_config(config: dict, path: str) -> None that saves a dict to JSON with indent=2 
Test: save a config dict, load it back, verify the values are identical
"""
import json
import logging

logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.error(e)
        return {}
       
        
def save_config(config: dict, path: str):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


path = "data/file.json"
sample_config = {"model": "gpt-4o-mini", "temperature": 0.2, "enabled": True}

save_config(sample_config, path)
config = load_config(path)

assert config == sample_config
assert load_config("missing_file.json") == {}