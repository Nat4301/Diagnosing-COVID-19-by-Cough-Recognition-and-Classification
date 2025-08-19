import os
import audio2numpy as a2n

def get_root_path_publicdata():
    path = 'D:\\EPQ Project Database\\public_dataset\\public_dataset\\' #Text is subject to change depending on file location
    return path

def get_root_path_df(status = 'positive'):
    root = 'D:\\EPQ Project Database\\{}\\'.format(status)
    return root

def get_time_path(covid):
    root = 'D:\\EPQ Project Database\\{}\\'.format(covid)
    return root

def get_all_file_names(root_path,filetype = '.txt'):
    Pure_files = []
    for root,dir,file in os.walk(root_path):
        for file_name in file:
            if file_name.endswith(filetype):
                Pure_files.append(file_name)
    return Pure_files

def get_all_path_name(root_path,file_names): #To nvaigate through different files in a given folder we use a FOR loop to check through all existent files
    full_file_paths = []
    for file_name in file_names:
        full_file_path = root_path + file_name
        full_file_paths.append(full_file_path)
    return full_file_paths

def get_cough_data(file_name):
    root = get_root_path_publicdata()
    full_path = root + file_name + '.wav'
    try: #Validation for file named
        amplitude,sample_rate = a2n.audio_from_file(full_path)
        return amplitude
    except FileNotFoundError:
        print("Error: File is not found")




if __name__ == "__main__":
    pass






# root = get_root_path()
# files = get_all_file_names(root)
# paths = get_all_path_name(root,files)
# print(paths)

# test = get_cough_data('ffd18a56-096d-40fc-9862-e5c5a8ca1fcd')
# print(test)