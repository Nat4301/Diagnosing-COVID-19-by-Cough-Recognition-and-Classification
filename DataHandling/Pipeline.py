
from DataHandling.CSVHandling import *
from MLClassifier.Covid_Classfication import *
import random


def get_root_path():
    path = 'D:\\Trained Classifier\\'  # Text is subject to change depending on file location
    return path


#This script will provide all the final versions of the functions and function calls I use

def load_time_series_data(covid):
    root = get_time_path(covid)
    names = get_covid_names(covid)
    number = random.randint(0,len(names))
    num2 = 2
    name = names[num2] ##Only loads first file in covid names later on we can just loop this function
    path = root + name + '.txt'
    df = load_csv_file(path)
    return df


