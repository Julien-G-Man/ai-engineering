import json
import json

config =  {
    'model': 'gbr',
    'n_estimators': 200,
    'features': ['dist', 'weight']
}

def dump(config):
    return json.dumps(config, indent=2)
   
def load(json_string):
    return json.loads(json_string)

def write_json_file(file):
    with open(file, 'w') as f:
       return json.dump(config, f, indent=2)
    
def read_json_file(file):
    with open(file, 'r') as f:
        return json.load(f)

    
json_string = dump(config)
loaded = load(json_string) 

if __name__ == "__main__":
    print(write_json_file('data/config.json'))
    print(read_json_file('data/config.json'))
    print(json_string, type(json_string))
    print(loaded, type(loaded))