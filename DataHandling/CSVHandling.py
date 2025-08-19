import pandas as pd
import pyarrow as pa
from pyarrow import csv
from Code.DataHandling.FileHandling import *
from Code.DataHandling.DFHandling import *
from Code.Utils.OS_Check import *


def get_metadata_path():
    root = get_root_pathfro_publicdata()
    MetaPath = root + 'metadata_compiled.csv'
    return MetaPath

def get_metadata():
    path = get_metadata_path()
    metadata = pd.read_csv(path) #metadata is a DF and CSV files are divided into rows and colums
    return metadata

def get_covid_positive():
    metadata = get_metadata()
    arr = []
    x = 0
    while x < 27550:
        if metadata.iloc[x,10] == "COVID-19" and metadata.iloc[x,2] > 0.5:
            arr.append(metadata.iloc[x,0])
        x += 1
    return arr

def get_covid_negative():
    metadata = get_metadata()
    arr = []
    x = 0
    while x < 27550:
        if metadata.iloc[x, 10] == "healthy" and metadata.iloc[x, 2] > 0.5:
            arr.append(metadata.iloc[x, 0])
        x += 1
    return arr

def get_covid_names(covid):
    if covid == 'positive':
        return get_covid_positive()
    else:
        return get_covid_negative()

def save_csv_pa(df,dir,filename):
    check_and_create_dir(dir)
    full_path = dir + filename
    df_pa_table = pa.Table.from_pandas(df)
    csv.write_csv(df_pa_table, full_path)
    print('File Saved')

def save_audiodata(status = "positive"): # = "Positive" Means a default peramter
    dir = get_time_path(status)
    names = get_covid_names(status)
    count = 0
    ccount = 0
    for name in names:
        print(count)
        data = get_cough_data(name).tolist()
        dict = {'Amplitude' : data}
        df = pd.DataFrame(dict)
        try:
            save_csv_pa(df,dir,name + '.txt')
            count += 1
        except:
            ccount += 1
            pass
    print('Corrupted Files',ccount,'.')

def load_csv_file(path): ## Important funciton gets the file data
    df_pa_1 = csv.read_csv(path)
    df_pa_1 = df_pa_1.to_pandas()
    return df_pa_1

def over_write_remove_zero(covid):
    root = get_time_path(covid)
    names = get_covid_names(covid)
    count = 0
    for name in names:
        try:
            path = root + name + '.txt'
            df = load_csv_file(path)
            df = get_remove_zero_df(df)
            save_csv_pa(df, root, name + '.txt')
        except:
            count += 1
            print("Save Error:",count,'.')


def save_compress_segment_df(covid):
    root = get_time_path(covid)
    names = get_covid_names(covid)
    save_path = "D:\\EPQ Project Database\\{} - compress\\".format(covid)
    count = 0
    count2 = 0
    for name in names:
        try:
            path = root + name + '.txt'
            df = load_csv_file(path)
            df = get_fully_segment(df,True,True)
            dict = {'Amplitude': df}
            df = pd.DataFrame(dict)
            save_csv_pa(df,save_path,name + '.txt')
            print("success")
            count += 1
            print("save count:",count,'.')
        except:
            count2 += 1
            print("fail count:",count2,'.')
            print("_________________________________")

if __name__ == "__main__":
    data = load_csv_file("D:\\EPQ Project Database\\positive - compress\\658daf16-7fc3-4640-911e-81dac60a2df9.txt")
    print(np.array(data['Amplitude']))


#Basic Function in Pandas:
# DF.head() : Returns the first 5 numbners of rows and colums
# DF.columns : Returns the names of the colums
# DF[name] : Returns all the values in the column
# DF.iloc[n,d] : Returns all the values in the row n position d
# DF.loc[(DF[ColumnName] == value) & (Condition like first one)] : Returns all the rows number of the satisfying condition

