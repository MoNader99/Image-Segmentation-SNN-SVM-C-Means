from PyQt5 import QtWidgets , QtCore
from pyqtgraph import PlotWidget, plot, PlotItem
from Classification import Ui_MainWindow
from skimage.transform import resize
from NeuralNetwork import MyNN
from sklearn import svm
from skimage import io
import pyqtgraph as pg
import pandas as pd
import numpy as np
import warnings
import pickle 
import math
import sys
import os
import cv2 


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        pg.setConfigOption('background', 'w')
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        warnings.filterwarnings("ignore")

        # ----------- C means -----------------
        self.img_array_Cmeans_original = []
        self.img_array_Cmeans_BW = []
        self.Segmented_Image = []
        self.clusterNumber = 1
        self.maximum_iteration = 500
        self.epsilon = 0.05
        self.m = 2   # fuzziness index which is usually set to 2
        self.flatten =  []
        self.pixelNum = []
        self.membership_degree = []
        # ------------------------------------

        # -------------- SNN -----------------
        self.img_array_SNN_original = []
        self.Segmented_Image_SNN = []
        # -----------------------------------

        # -------------- SVM -----------------
        self.img_array_SMV_original = []
        self.Segmented_Image_SVM = []
        # -----------------------------------

        self.graphic_View_Array=[self.ui.OriginalImage,self.ui.ClusterdImage,self.ui.OriginalImage_SVM,self.ui.ClusterdImage_SVM,self.ui.OriginalImage_SNN,self.ui.ClusterdImage_SNN]
        for x in self.graphic_View_Array:
            x.getPlotItem().hideAxis('bottom')
            x.getPlotItem().hideAxis('left')
            x.setMouseEnabled(x=False, y=False)

        self.ui.StartButton.clicked.connect(self.start)
        self.ui.ImportButton.clicked.connect(self.Import)
        self.ui.Cluster_Slider.valueChanged.connect(self.sliderChange)

        self.ui.ImportButtonSVM.clicked.connect(self.Import_SVM)
        self.ui.ShowResult.clicked.connect(self.showResults)

        self.ui.ImportButtonSNN.clicked.connect(self.Import_SNN)
        self.ui.ShowResult_SNN.clicked.connect(self.showResults_SNN)


# ----------------------------------- utilties functions ------------------------------------------------
    def sliderChange(self):
        number_of_clusters = self.ui.Cluster_Slider.value()
        self.clusterNumber = number_of_clusters
        self.ui.Cluster_Label.setText(str(number_of_clusters))
        return
    
    def plot(self,image,GV):
        ploted_image = pg.ImageItem(image)
        GV.clear()
        GV.addItem(ploted_image) 
        ploted_image.rotate(270)
# ------------------------------------------------------------------------------------------------------

# --------------------------------------- SNN functions ------------------------------------------------
    def Import_SNN(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "Desktop", '*')
        if fname:
            self.img_array_SNN_original = io.imread(fname)
            # self.img_array_SMV_original = cv2.resize(self.img_array_SMV_original,(300,300), interpolation = cv2.INTER_AREA)
            self.plot(self.img_array_SNN_original,self.ui.OriginalImage_SNN)
        return

    def showResults_SNN(self):
        if len(self.img_array_SNN_original) != 0 :
            x_test = self.img_array_SNN_original
            pixel_values = np.float32(x_test)/255
            x_test = np.array(pixel_values)
            testclass = pd.DataFrame(columns = ['R','G','B'])
            testclass = testclass.append(pd.DataFrame(x_test.reshape(-1,3),columns = ['R','G','B']))
            x_test = testclass.iloc[:,:].values

            file = open(os.path.join(sys.path[0],'SNN_Model_Pickle'),'rb')
            model = pickle.load(file)
            # model = pickle.load()
            
            y_pred = model.predict(x_test)
            y_pred = np.array(y_pred)

            self.Segmented_Image_SNN = self.GetSegmentedPhoto_SNN(y_pred,x_test)

            self.plot(self.Segmented_Image_SNN,self.ui.ClusterdImage_SNN)
        else:
            return

    def GetSegmentedPhoto_SNN(self,y,x):
        cluster_mean  = []
        cluster_mean2 = []
        cluster_mean3 = []

        for i in range(len(x)):
            if y[i] == 0:
                cluster_mean.append(x[i])
            if y[i] == 1:
                cluster_mean2.append(x[i])
            if y[i] == 2:
                cluster_mean3.append(x[i])
        
        cluster_mean = np.mean(cluster_mean,axis=0)
        cluster_mean2 = np.mean(cluster_mean2,axis=0)
        cluster_mean3 = np.mean(cluster_mean3,axis=0)
        y.reshape(-1,1)
        y_mean = []
        for i in y:
            if i == 0:
                y_mean.append(cluster_mean)
            if i == 1:
                y_mean.append(cluster_mean2)
            if i == 2:
                y_mean.append(cluster_mean3)
                
        y_mean = np.array(y_mean)
        reshape_vale=int(math.sqrt(y.shape[0]))

        return(y_mean.reshape(reshape_vale,reshape_vale,3))

# ------------------------------------------------------------------------------------------------------

# --------------------------------------- SVM functions ------------------------------------------------
    def Import_SVM(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "Desktop", '*')
        if fname:
            self.img_array_SMV_original = io.imread(fname)
            # self.img_array_SMV_original = cv2.resize(self.img_array_SMV_original,(300,300), interpolation = cv2.INTER_AREA)
            self.plot(self.img_array_SMV_original,self.ui.OriginalImage_SVM)
        return

    def showResults(self):
        if len(self.img_array_SMV_original) != 0 :
            x_test = self.img_array_SMV_original
            pixel_values = np.float32(x_test)/255
            x_test = np.array(pixel_values)
            testclass = pd.DataFrame(columns = ['R','G','B'])
            testclass = testclass.append(pd.DataFrame(x_test.reshape(-1,3),columns = ['R','G','B']))
            x_test = testclass.iloc[:,:].values

            with open(os.path.join(sys.path[0],'SVM_Model_Pickle') ,'rb') as f:
                model = pickle.load(f)
            
            y_pred = model.predict(x_test)
            
            self.Segmented_Image_SVM = self.GetSegmentedPhoto(y_pred,x_test)

            self.plot(self.Segmented_Image_SVM,self.ui.ClusterdImage_SVM)
        else:
            return

    def GetSegmentedPhoto(self,y,x):
        cluster_mean  = []
        cluster_mean2 = []
        cluster_mean3 = []

        for i in range(len(x)):
            if y[i] == 1:
                cluster_mean.append(x[i])
            if y[i] == 2:
                cluster_mean2.append(x[i])
            if y[i] == 3:
                cluster_mean3.append(x[i])
        
        cluster_mean = np.mean(cluster_mean,axis=0)
        cluster_mean2 = np.mean(cluster_mean2,axis=0)
        cluster_mean3 = np.mean(cluster_mean3,axis=0)
        y.reshape(-1,1)
        y_mean = []
        for i in y:
            if i == 1:
                y_mean.append(cluster_mean)
            if i == 2:
                y_mean.append(cluster_mean2)
            if i == 3:
                y_mean.append(cluster_mean3)
                
        y_mean = np.array(y_mean)
        reshape_vale=int(math.sqrt(y.shape[0]))

        return(y_mean.reshape(reshape_vale,reshape_vale,3))

# ----------------------------------------------------------------------------------------------------------
        
# ---------------------------------------- c means functions -----------------------------------------------
    def Import(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "Desktop", '*')
        if fname:
            self.img_array_Cmeans_original = cv2.imread(fname) #cv2.resize(cv2.imread(fname), (64,64), interpolation = cv2.INTER_AREA)
            self.img_array_Cmeans_BW = cv2.cvtColor(self.img_array_Cmeans_original, cv2.COLOR_BGR2GRAY)
            self.flatten =  self.img_array_Cmeans_BW.flatten().astype('float')
            self.pixelNum = self.img_array_Cmeans_BW.size
        return

    def start(self):
        if len(self.img_array_Cmeans_original) != 0:
            self.membership_degree = np.zeros((self.pixelNum, self.clusterNumber))
            idx = np.arange(self.pixelNum)
            for cluster in range(self.clusterNumber):
                idxii = idx%self.clusterNumber==cluster
                self.membership_degree[idxii,cluster] = 1

            self.Cluster_center = np.linspace(np.min(self.img_array_Cmeans_BW),np.max(self.img_array_Cmeans_BW),self.clusterNumber)
            self.Cluster_center = self.Cluster_center.reshape(self.clusterNumber,1)
            self.c_mean_operation()

            self.plot(cv2.cvtColor(self.img_array_Cmeans_original, cv2.COLOR_BGR2RGB),self.ui.OriginalImage)
            self.plot(self.Segmented_Image,self.ui.ClusterdImage)

        else:
            return

    def update_membershipDegree(self):
        c_mesh,x_mesh = np.meshgrid(self.Cluster_center,self.flatten)
        power = 2./(self.m-1)
        p1 = abs(x_mesh-c_mesh)**power  # Ecliden distance between cluster center and data
        p2 = np.sum((1./p1),axis=1)
        return 1./(p1*p2[:,None])

    def update_Centers(self):
        num = np.dot(self.flatten,self.membership_degree**self.m)
        den = np.sum(self.membership_degree**self.m,axis=0)
        return num/den

    def c_mean_operation(self):
        i = 0
        while True:
            self.Cluster_center = self.update_Centers()
            old_degree = np.copy(self.membership_degree)
            self.membership_degree = self.update_membershipDegree()
            difference = np.sum(abs(self.membership_degree - old_degree))
            printed = str(i) + " - difference = " + str(format(difference, '.4f'))
            self.ui.IterationsLable.setText(printed)
            
            if difference < self.epsilon or i > self.maximum_iteration:
                break
            i += 1
        self.Segmented_Image = np.argmax(self.membership_degree, axis=1)
        self.Segmented_Image = self.GetSegmentedImage_Cmeans(self.Segmented_Image,cv2.cvtColor(self.img_array_Cmeans_original, cv2.COLOR_BGR2RGB))

    # i hope no one see this code :)
    def GetSegmentedImage_Cmeans(self,y,x):
        pixel_values = np.float32(x)/255
        x_test = np.array(pixel_values)
        testclass = pd.DataFrame(columns = ['R','G','B'])
        testclass = testclass.append(pd.DataFrame(x_test.reshape(-1,3),columns = ['R','G','B']))
        x_test = testclass.iloc[:,:].values
        segmented = []
        number_of_clusters = self.ui.Cluster_Slider.value()
        array = [i  for i in range(number_of_clusters)]
        means = []
        for i in range(number_of_clusters):
            temp = []
            means.append(temp)
        for i in range(number_of_clusters):
            means[i].append(np.mean(x_test[y == array[i]],axis=0))

        for j in range(len(y)):
            if y[j] == 0:
                segmented.append(means[0][0])
            if y[j] == 1:
                segmented.append(means[1][0])
            if number_of_clusters == 3:
                if y[j] == 2:
                    segmented.append(means[2][0])
            if number_of_clusters == 4:
                if y[j] == 2:
                    segmented.append(means[2][0])
                if y[j] == 3:
                    segmented.append(means[3][0])
            if number_of_clusters == 5:
                if y[j] == 2:
                    segmented.append(means[2][0])
                if y[j] == 3:
                    segmented.append(means[3][0])
                if y[j] == 4:
                    segmented.append(means[4][0])
            if number_of_clusters == 6:
                if y[j] == 2:
                    segmented.append(means[2][0])
                if y[j] == 3:
                    segmented.append(means[3][0])
                if y[j] == 4:
                    segmented.append(means[4][0])
                if y[j] == 5:
                    segmented.append(means[5][0])
            if number_of_clusters == 7:
                if y[j] == 2:
                    segmented.append(means[2][0])
                if y[j] == 3:
                    segmented.append(means[3][0])
                if y[j] == 4:
                    segmented.append(means[4][0])
                if y[j] == 5:
                    segmented.append(means[5][0])
                if y[j] == 6:
                    segmented.append(means[6][0])
            if number_of_clusters == 8:
                if y[j] == 2:
                    segmented.append(means[2][0])
                if y[j] == 3:
                    segmented.append(means[3][0])
                if y[j] == 4:
                    segmented.append(means[4][0])
                if y[j] == 5:
                    segmented.append(means[5][0])
                if y[j] == 6:
                    segmented.append(means[6][0])
                if y[j] == 7:
                    segmented.append(means[7][0])
            if number_of_clusters == 9:
                if y[j] == 2:
                    segmented.append(means[2][0])
                if y[j] == 3:
                    segmented.append(means[3][0])
                if y[j] == 4:
                    segmented.append(means[4][0])
                if y[j] == 5:
                    segmented.append(means[5][0])
                if y[j] == 6:
                    segmented.append(means[6][0])
                if y[j] == 7:
                    segmented.append(means[7][0])
                if y[j] == 8:
                    segmented.append(means[8][0])
            if number_of_clusters == 10:
                if y[j] == 2:
                    segmented.append(means[2][0])
                if y[j] == 3:
                    segmented.append(means[3][0])
                if y[j] == 4:
                    segmented.append(means[4][0])
                if y[j] == 5:
                    segmented.append(means[5][0])
                if y[j] == 6:
                    segmented.append(means[6][0])
                if y[j] == 7:
                    segmented.append(means[7][0])
                if y[j] == 8:
                    segmented.append(means[8][0])
                if y[j] == 9:
                    segmented.append(means[9][0])            
        segmented = np.array(segmented)
        reshape_vale=int(math.sqrt(segmented.shape[0]))
        return(segmented.reshape(reshape_vale,reshape_vale,3))
# ------------------------------------------------------------------------------------------------------


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()

if __name__ == "__main__":
    main()

# https://www.baeldung.com/cs/svm-multiclass-classification