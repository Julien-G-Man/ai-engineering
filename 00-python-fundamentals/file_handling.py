from pathlib import Path

data_dir = Path('data/raw')
data_dir.mkdir(parents=True, exist_ok=True) # Create dirs safely 

csv_path = data_dir / 'logistics.csv'          # '/' joins paths cleanly 
print(csv_path)          # data/raw/logistics.csv 
print(csv_path.exists()) # True or False 
print(csv_path.stem)     # 'logistics'  (filename without extension) 
print(csv_path.suffix)   # '.csv' 