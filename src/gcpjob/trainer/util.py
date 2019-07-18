import pandas as pd
#import os
#from tensorflow.python.lib.io import file_io
#import matplotlib.pyplot as plt

classDf = pd.DataFrame({'Class': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        'ClassName': ['Air Conditioner', 'Car Horn', 'Children Playing',
                                      'Dog Bark', 'Drilling', 'Engine Idling',
                                      'Gun Shot', 'Jackhammer', 'Siren', 'Street Music']})
classNames = classDf['ClassName']

# def copy_file_to_gcs(job_dir, file_path):
#   with file_io.FileIO(file_path, mode='rb') as input_f:
#     with file_io.FileIO(
#         os.path.join(job_dir, file_path), mode='w+') as output_f:
#       output_f.write(input_f.read())