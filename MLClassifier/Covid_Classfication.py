import matplotlib.pyplot as plt
import pandas as pd

from DataHandling.CSVHandling import *
from DataHandling.FileHandling import *
from Maths.Cleaning import *
import numpy as np
import time as tm

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

import pickle as pk



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


class CovidClassifier():
    def __init__(self):
        pass

    def get_root_path(self,covid):
        rootpath = "D:\\EPQ Project Database\\{} - compress\\".format(covid)
        return rootpath

    def load_data(self,full_path):
        data = load_csv_file(full_path)
        return np.array(data['Amplitude']) #Returns the data not the labels

    def get_full_file_path(self,covid):
        rootpath = self.get_root_path(covid)
        file_names = get_all_file_names(rootpath)
        full_file_paths = get_all_path_name(rootpath, file_names)
        return full_file_paths

    def make_clean_array(self,method = 'Resize'): #Method = Fixed,Sum,Gradient,Fourier
        covid_status = ['positive','negative']
        data_list = []
        status_list = []
        size = 0
        for status in covid_status:
            full_path = self.get_full_file_path(status)

            if status == 'positive': #For testing purposes
                size = len(full_path)
            else:
                full_path = full_path[:4750]#4750 resulted in the highest degree of accuracy
            print(len(full_path))



            #full_path = [full_path[-1]]
            # full_path = ["D:\\EPQ Project Database\\negative - compress\\1ed72aa1-60f1-4a7a-b300-98d49ed455bb.txt"]
            # print(full_path)

            for path in full_path:
                data = self.load_data(path)
                if method == "Resize":
                    data = get_rezize_same_lenth(data)

                else:
                    print('Error method not valid')
                data_list.append(data)
                if status == 'positive':
                    status_list.append('1')
                else:
                    status_list.append('0')

        #Makes a clean df for our ML model
        clean_df = pd.DataFrame()
        for idx,data in enumerate(data_list):
            clean_df[idx] = data

        #transpose the dataframe
        clean_df = clean_df.T

        #asign feautre names
        clean_df.columns = self.get_feature_name(data_list)
        self.feautre_names = self.get_feature_name(data_list)

        #Assign the status
        clean_df['status'] = status_list
        self.DF = clean_df

        #Data frame complete

    def get_feature_name(self,list):
        col = []
        for i in range(len(list[0])):
            col.append('Amp{}'.format(i))
        return col

    def split_data(self,testsize):

        self.train_features,self.test_features,self.train_labels,self.test_labels = train_test_split(self.DF[self.feautre_names],self.DF['status'],test_size=testsize,shuffle=True)

        print(self.test_features)
        print(self.test_labels)

        print('Compression Complete')

    def KNeighbors_model(self): #0 for negative, #1 for positive:
        st = tm.time()
        clf = KNeighborsClassifier()
        Kmodel_fit = clf.fit(self.train_features,self.train_labels)
        prediction = np.array(Kmodel_fit.predict(self.test_features))
        et = tm.time()
        # print(prediction)

        test1 = np.array(self.test_labels)
        print('KNeighbors error:',accuracy_score(test1, prediction))
        print('Wall Elapsed Time:',et - st)
        return accuracy_score(prediction, test1), et - st
        # error = mean_squared_error(test1,prediction)

    def Decision_tree_model(self):
        st = tm.time()
        clf = tree.DecisionTreeClassifier()
        Tmodel_fit = clf.fit(self.train_features,self.train_labels)
        prediction = np.array(Tmodel_fit.predict(self.test_features))
        et = tm.time()
        # print(prediction)
        test1 = np.array(self.test_labels)
        print('Decision Tree Error:',accuracy_score(test1, prediction))
        print('Wall Elapsed Time:', et - st)
        return accuracy_score(prediction, test1), et - st
        # error = mean_squared_error(test1, prediction)


    def Naive_Bayes_model(self):
        st = tm.time()
        clf = GaussianNB()
        Nmodel_fit = clf.fit(self.train_features,self.train_labels)
        prediction = np.array(Nmodel_fit.predict(self.test_features))
        et = tm.time()
        # print(prediction)
        test1 = np.array(self.test_labels)
        print('Naive Bayes Error:',accuracy_score(test1, prediction))
        print('Wall Elapsed Time:', et - st)
        return accuracy_score(prediction, test1), et - st
        # error = mean_squared_error(test1, prediction)


    def Linear_Discriminant_model(self):
        st = tm.time()
        clf = LinearDiscriminantAnalysis()
        Lmodel_fit = clf.fit(self.train_features,self.train_labels)
        prediction = np.array(Lmodel_fit.predict(self.test_features))
        et = tm.time()
        # print(prediction)
        test1 = np.array(self.test_labels)
        print('Linear Discrminant Error:',accuracy_score(test1, prediction))
        print('Wall Elapsed Time:', et - st)
        return accuracy_score(prediction, test1), et - st
        # error = mean_squared_error(test1, prediction)


    def SVC_model(self):
        st = tm.time()
        clf = SVC()
        Smodel_fit = clf.fit(self.train_features,self.train_labels)
        prediction = np.array(Smodel_fit.predict(self.test_features))
        et = tm.time()
        # print(prediction)
        test1 = np.array(self.test_labels)
        print('SVC Error:', accuracy_score(prediction, test1))
        print('Wall Elapsed Time:', et - st)
        return accuracy_score(prediction, test1), et - st
        # error = mean_squared_error(test1, prediction)


    def Logistic_Regression_model(self):
        st = tm.time()
        clf = LogisticRegression()
        Rmodel_fit = clf.fit(self.train_features,self.train_labels)
        prediction = np.array(Rmodel_fit.predict(self.test_features))
        et = tm.time()
        # print(prediction)
        test1 = np.array(self.test_labels)
        print('Logistic Regression Error:', accuracy_score(test1, prediction))
        print('Wall Elapsed Time:', et - st)
        return accuracy_score(prediction,test1), et - st


        # error = mean_squared_error(test1, prediction)


    def Ridge_Regression_model(self):
        st = tm.time()
        clf = RidgeClassifier(alpha=1e-3)
        RCmodel_fit = clf.fit(self.train_features,self.train_labels)
        prediction = np.array(RCmodel_fit.predict(self.test_features))
        et = tm.time()
        # print(prediction)
        test1 = np.array(self.test_labels)
        print('Ridge Regression Error:', accuracy_score(test1, prediction))
        print('Wall Elapsed Time:', et - st)
        return accuracy_score(prediction,test1), et - st

    def Random_Forest_model(self):
        st = tm.time()
        clf = RandomForestClassifier()
        RFmodel_fit = clf.fit(self.train_features,self.train_labels)
        prediction = np.array(RFmodel_fit.predict(self.test_features))
        et = tm.time()
        # print(prediction)
        test1 = np.array(self.test_labels)
        print('Random Forest Error:', accuracy_score(test1, prediction))
        print('Wall Elapsed Time:', et - st)
        return accuracy_score(prediction,test1), et - st

    def MLP_classifier_model(self):
        st = tm.time()
        clf = MLPClassifier()
        RCmodel_fit = clf.fit(self.train_features,self.train_labels)
        prediction = np.array(RCmodel_fit.predict(self.test_features))
        et = tm.time()
        # print(prediction)
        test1 = np.array(self.test_labels)
        print('MLP Error:', accuracy_score(test1, prediction))
        print('Wall Elapsed Time:', et - st)
        return accuracy_score(prediction,test1), et - st

    def SGD_classifier_model(self):
        st = tm.time()
        clf = SGDClassifier()
        RCmodel_fit = clf.fit(self.train_features,self.train_labels)
        prediction = np.array(RCmodel_fit.predict(self.test_features))
        et = tm.time()
        # print(prediction)
        test1 = np.array(self.test_labels)
        print('SGD Error:', accuracy_score(test1, prediction))
        print('Wall Elapsed Time:', et - st)
        return accuracy_score(prediction,test1), et - st

    def Model_testing(self,split = 0.3,number = 1):
        self.make_clean_array()
        self.split_data(split)
        models = ['Kmodel','Dmodel','Nmodel','LDmodel','LRmodel','RCmodel','RFModel','MLPModel','SGDmodel']
        models2 = ['Kmodel','Nmodel','RFModel','SGDmodel']
        accuracys = []
        times = []
        for count in range(number):
            temp_accuracys = []
            temp_times = []

            for idx,model in enumerate(models):
                if idx == 0:
                    accuracy,time = self.KNeighbors_model()
                # elif idx == 1:
                #     accuracy,time = self.Decision_tree_model()
                elif idx == 2:
                    accuracy,time = self.Naive_Bayes_model()
                # elif idx == 3:
                #     accuracy,time = self.Linear_Discriminant_model()
                # elif idx == 4:
                #     accuracy,time = self.Logistic_Regression_model()
                # elif idx == 5:
                #     accuracy,time = self.Ridge_Regression_model()
                elif idx == 6:
                    accuracy,time = self.Random_Forest_model()
                # elif idx == 7:
                #     accuracy,time = self.MLP_classifier_model()
                elif idx == 8:
                    accuracy,time = self.SGD_classifier_model()

                temp_accuracys.append(accuracy)
                temp_times.append(time)
            accuracys.append(temp_accuracys)
            times.append(temp_times)

        a_accuracys = []
        a_times = []
        for idx in range(len(models2)):
            a_accuracys.append(self.get_average(accuracys,idx))
            a_times.append(self.get_average(times, idx))


        plt.figure()
        plt.bar(np.arange(len(models2)),a_accuracys,width=0.5,label = 'Accuracy')
        plt.xticks(np.arange(len(models2)),models)
        plt.figure()
        plt.bar(np.arange(len(models2)), a_times, width=0.5, label = 'Time')
        plt.xticks(np.arange(len(models2)), models)
        plt.show()
        plt.plot(accuracy)


    def get_average(self,matrix,index):
        a_matrix = [row[index] for row in matrix]
        return np.average(a_matrix)

    def save_model(self,model,test_size):
        fpath = 'D:\\Trained Classifier\\'
        fname = fpath + 'Trained_{}'.format(model) + '.sav'
        self.make_clean_array()
        self.split_data(test_size)
        clf = RandomForestClassifier()
        clf.fit(self.train_features, self.train_labels)
        mlfile = open(fname,"wb")
        pk.dump(clf,mlfile)
        mlfile.close()

        print("Save Complete")
        mlfile = open(fname,'rb')
        loadedmodel = pk.load(mlfile)
        prediction = loadedmodel.predict(self.test_features)
        print(prediction)
        print('KNeighbors error:', accuracy_score(self.test_labels, prediction))


if __name__ == '__main__':
    test1 = CovidClassifier()
    test1.save_model('RFmodel',0.2)



