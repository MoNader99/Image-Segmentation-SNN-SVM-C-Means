import numpy as np 
import cv2
import sys
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename

class FCM ():
    def __init__ (self,image,epsilon,clusterNumber,maximum_iteration):
        self.seg = [0]
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.originalImage = image
        self.clusterNumber = clusterNumber
        self.maximum_iteration = maximum_iteration
        self.epsilon = epsilon
        self.m = 2   # fuzziness index which is usually set to 2
        self.flatten =  self.image.flatten().astype('float')
        self.pixelNum = self.image.size

        # -------------------  membership_degree intialization -----------------------------

        self.membership_degree = np.zeros((self.pixelNum, self.clusterNumber))
        idx = np.arange(self.pixelNum)
        for cluster in range(self.clusterNumber):
            idxii = idx%self.clusterNumber==cluster
            self.membership_degree[idxii,cluster] = 1

        # -------------------  membership_degree intialization -----------------------------
        
        # -------------------  cluster center intialization -----------------------------

        self.Cluster_center = np.linspace(np.min(self.image),np.max(self.image),clusterNumber)
        self.Cluster_center = self.Cluster_center.reshape(self.clusterNumber,1)

        # -------------------  cluster center intialization -----------------------------

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
        # Repeat until convergence
        i = 0
        while True:
            # Compute centroid for each cluster
            self.Cluster_center = self.update_Centers()

            # Save initial membership matrix
            old_degree = np.copy(self.membership_degree)

            # Update coefficients for each pixel
            self.membership_degree = self.update_membershipDegree()

            # Difference between initial mem matrix and new one
            difference = np.sum(abs(self.membership_degree - old_degree))
            print(str(i) + " - difference = " + str(difference))

            # Check convergence
            if difference < self.epsilon or i > self.maximum_iteration:
                break
            i += 1

        self.seg = np.argmax(self.membership_degree, axis=1)
        self.seg = self.seg.reshape(self.image.shape).astype('int')
        

    def plot(self):
        # figsize=(8, 4), dpi=100
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(cv2.cvtColor(self.originalImage, cv2.COLOR_BGR2RGB))
        ax1.set_title('original')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(self.seg)
        ax2.set_title('segmentation')

        plt.show()

if __name__ == "__main__":
    filename = askopenfilename()
    image = cv2.imread(filename)
    fuz = FCM(image,0.05 , 3, 500)
    fuz.c_mean_operation()
    fuz.plot()

