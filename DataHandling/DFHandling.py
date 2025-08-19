from Code.DataHandling.CSVHandling import *
from Code.DataHandling.FileHandling import *
from Code.Maths.Compression import *

import pandas as pd
import pyarrow as pa
import numpy as np

def get_clean_df(covid):
    root = get_root_path_df(covid)
    arr = get_covid_names(covid)
    clean = pd.DataFrame({})

    for name in arr:
        path = root + name + '.txt'
        try:
            df = load_csv_file(path)
            clean = pd.concat([clean,df])
        except:
            pass
    return clean


def get_remove_value_df(df,value):
    arr = np.array(df)
    clean_arr = remove_values_less_than(arr,value)
    dict = {'amplitude' : clean_arr}
    clean_df = pd.DataFrame(dict)
    return clean_df

def get_fully_segment(df,normal = True,remove_cond = True): #Segments and Compress signal
    arr = np.array(df)
    segment_arr = segment_signal(arr)
    if segment_arr != False:

        if normal == True:
            segment_arr = normalise_list(segment_arr)
        if remove_cond ==  True:
            segment_arr = remove_values_two_conditions(segment_arr)

    else:
        return False
    return segment_arr

def get_data_cleaning_df(covid,fixed = False, sum = False,
                         Gradient = False,Fourier = False):
    save_path = "D:\\EPQ Project Database\\{} - compress\\".format(covid)
    pass


if __name__ == '__main__':
    df = get_clean_df('positive')
    print(df)

