import numpy as np
import pdb, glob, time, os, imageio, copy, cv2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from ripser import ripser
from persim import plot_diagrams
import gudhi as gd
from scipy.stats import multivariate_normal
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf') 
import matplotlib as mpl
from sklearn import svm
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
    
def clustering_fine_train_test(X,chi_range,rho_range,real_range,num_clusters=8,filename=None,title=None):

	'''
	
	Clusters data from X based on training data, and then test how consistent these 
	clusters are for out-of-sample data

	inputs 
	
	X.     		 : all data being used for clustering & classification 
	chi_range 	 : labels the chi value used for each row of X
	rho_range    : labels the rho value used for each row of X
	real_range   : labels the realization for each (rho,chi) combinations for each row of X
				  used to split data into test/train split
	num_clusters : number of clusters used during clustering
	filename     : how to save figures

	outputs

	score        : out of sample accuracy of kmeans prediction

	'''

	#split data into testing, training splits
	X_train = X[real_range<7,:]
	X_test = X[real_range>=7,:]

	rho_train = rho_range[real_range<7]
	chi_train = chi_range[real_range<7]

	rho_test = rho_range[real_range>=7]
	chi_test = chi_range[real_range>=7]

	#perform k-means clustering on the training data
	kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X_train)
    
	#retrieve labels to label the training data
	kmeans_class = kmeans.labels_
	
	#find unique chi,rho values considered
	chi_vec = np.unique(chi_range)
	rho_vec = np.unique(rho_range)

	#initialize labels for each (chi,rho) location
	common_label_train = np.zeros((len(rho_vec),len(chi_vec)))

	#determine training labels based on classification of (chi,rho) region
	for j,chi in enumerate(chi_vec):
		for i,rho in enumerate(rho_vec):

			chi_rho_range = np.where(np.logical_and(chi_train==chi,rho_train==rho))[0]
			chi_rho_labels = kmeans_class[chi_rho_range]

			counts = np.bincount(chi_rho_labels)
			common_label_train[i,j] = np.argmax(counts)

	### Redefine common_label_train for easier coloring
	CHI,RHO = np.meshgrid(np.unique(chi_vec),np.unique(rho_vec))
    
	chi_mean = np.zeros((num_clusters,))
	for i in np.arange(num_clusters):
		chi_mean[i] = np.mean(CHI[common_label_train==i])
    
	common_label_train_copy = np.zeros(CHI.shape)
    
	for iold, inew in enumerate(np.argsort(chi_mean)):
		common_label_train_copy[common_label_train == inew] = iold      
    
	#Now use these training labels to create the test labels
	Ytest = np.zeros(rho_test.shape)        
	Ytrain = np.zeros(rho_train.shape)        
	for j,chi in enumerate(chi_vec):
		for i,rho in enumerate(rho_vec):

			chi_rho_range = np.where(np.logical_and(chi_train==chi,rho_train==rho))[0]
			Ytrain[chi_rho_range] = common_label_train[i,j]

            
			chi_rho_range = np.where(np.logical_and(chi_test==chi,rho_test==rho))[0]
			Ytest[chi_rho_range] = common_label_train[i,j]
                        
    #generate predictions for test data
	kmeans_fit = kmeans.predict(X_train)
	kmeans_predict = kmeans.predict(X_test)
	centers = kmeans.cluster_centers_
    
	kmeans_predict_copy = np.zeros(kmeans_predict.shape)
	Ytest_copy = np.zeros(Ytest.shape)
    
	for iold, inew in enumerate(np.argsort(chi_mean)):
		kmeans_predict_copy[kmeans_predict == inew] = iold      
		Ytest_copy[Ytest == inew] = iold      
    
    #accuracy of the kmeans prediction
	acc = accuracy_score(Ytest_copy,kmeans_predict_copy)
	acc_in_sample = accuracy_score(Ytrain,kmeans_fit)
	#create confusion matrix
	cm = confusion_matrix(Ytest_copy,kmeans_predict_copy,labels=np.arange(num_clusters))
	#normalize
	cm_sum = cm.sum(axis=1)
	cm = cm / cm_sum[:,np.newaxis]
	cm[np.isnan(cm)] = 0
	if filename is not None:
		font = {'family' : 'normal','size'   : 20}

		plt.rc('font', **font)

		fig = plt.figure(figsize=(9,7))
		ax = fig.add_subplot(1,1,1)
		#plot confusion matrix
		plt.matshow(cm,cmap = plt.cm.Blues,vmin=0,vmax=1,fignum=0)
		if title is not None:
			plt.title(title + " confusion matrix,\n "+str(round(acc*100,1))+"% OOS Accuracy",fontsize=18,pad=-3)
		else:
			plt.title("Confusion Matrix, ("+str(round(acc*100,1))+"% OOS Accuracy) \n ")
        
		plt.ylabel(r"Group based on true $(\rho,\chi)$")
		plt.xlabel("Predicted group from k-means")
		ax.xaxis.tick_bottom()
		ax.set_xticklabels(np.arange(num_clusters+1))
		ax.set_yticklabels(np.arange(num_clusters+1))        
		plt.colorbar()

		for i in np.arange(num_clusters):
			for j in np.arange(num_clusters):
				if cm[j,i] < .8:
					plt.text(j,i, str(round(cm[i,j]*100,0))+"%", va='center', ha='center',color="black")
				else:
					plt.text(j,i, str(round(cm[i,j]*100,0))+"%", va='center', ha='center',color="white")

    
		plt.savefig("figures/"+filename+"_kmeans_CM.pdf",format="pdf")

		#colormap
		cmaplist = [(0.0,0.0,0.0),(0.0,0.0,1.0),(1.0,0.0,0.0),(1.0,1.0,0.0),(.5,.5,.5),(1.0,1.0,1.0),(0,1.0,1.0),(0.0,1.0,0.0)]
		cmaplist = cmaplist[:num_clusters]
		cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, N = num_clusters)
                
		fig = plt.figure(figsize=(9,7))
		ax = fig.add_subplot(1,1,1)
        
		#plot different classification areas in (chi,rho) space
		cax = plt.matshow(np.flipud(common_label_train_copy.T),cmap=cmap,fignum=0,vmin=0,vmax=num_clusters-1)
		#plt.scatter(CHI,RHO,c=common_label_train_copy.T,cmap=cmap)#,common_label_train_copy.T)#,cmap=cmap,fignum=0)
		ax.xaxis.set_label_position('bottom') 
		ax.xaxis.tick_bottom()
		ax.set_xticks([0,2,4,6,8,10])       
		ax.set_yticks([0,2,4,6,8,10])               
        
		ax.set_xticklabels(chi_vec[::2])
		ax.set_yticklabels(chi_vec[::-2])
        
		cticks = np.arange(0.5*((num_clusters-1)/num_clusters),num_clusters-1,(num_clusters-1)/num_clusters).tolist()        
        
		cbar = fig.colorbar(cax,ticks = cticks)
		cbar.ax.set_yticklabels(np.arange(1,num_clusters+1).tolist())
        
		plt.ylabel("Chemotaxis ($\chi$)")
		plt.xlabel(r"Haptotaxis ($\rho$)")
		if title is not None:
			plt.title(title + " clustering")
		else:
			plt.title("Parameter clustering, ("+str(round(acc*100,1))+"% OOS Accuracy)")

		plt.savefig("figures/"+filename+"_param_clustering.pdf",format="pdf")
	kmeans_class = kmeans.predict(X)    
			
	return kmeans_class,acc,acc_in_sample,centers

def create_latex_table_classification_sort(X_vec,title_list,chi_range,rho_range,real_range,num_clusters=8):
    acc = []
    acc_in_sample = []

    for X in X_vec:

        _,acc_tmp,acc_in_sample_tmp,centers = clustering_fine_train_test(X,chi_range,rho_range,
                                                        real_range,num_clusters=num_clusters)

        acc.append(acc_tmp)
        acc_in_sample.append(acc_in_sample_tmp)


    print("\\begin{tabular}{|c|c|c|}")
    print("\\hline")
    print("Feature & In Sample Accuracy & Out of Sample Accuracy \\\\ ")
    print("\\hline")
    
    for i in np.argsort(acc)[::-1]:
    
        print(title_list[i] + " & " + str(round(acc_in_sample[i]*100,1)) + "\% & " + str(round(acc[i]*100,1)) + "\% \\\\")
        print("\\hline")

    print("\\end{tabular}")
    print("\\caption{Accuracy sample scores for various feature vectors using $k$-means classification. LTR: Left to right topology, RTL: Right to left topology, TTB: Top to bottom topology, BTT: Bottom to top topology, PIR: Persistence image with ramp weighting, PIO: Persistence image with one weighting, BC: Betti Curve.}")