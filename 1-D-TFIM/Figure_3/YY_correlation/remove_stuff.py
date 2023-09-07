import os
import shutil

for folder in os.listdir('correlation_folder'):  
     
    if os.path.isdir(os.path.join('correlation_folder', folder)):  
        print(folder)                        
        for file in os.listdir(os.path.join('correlation_folder', folder)):
            if os.path.isdir(os.path.join('correlation_folder', folder, file)): 
                for f in os.listdir(os.path.join('correlation_folder', folder, file)):
                    #print(file)
                    if f.endswith('dicts'):
                        print(f)
                        shutil.rmtree(os.path.join('correlation_folder', folder, file, f))