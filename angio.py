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


def grad_est(n,x_n,y_n,C):
    
	'''
	Estimates the gradient of the matrix C at point n with respect to x and y

	inputs:
	
	n 		: point where we are estimating the gradient
	x_n 	: max x-value
	y_n 	: max y-value
	C 		: Matrix under consideration

	outputs:

	x_grad 		: centered difference in x (unless boundary point)
	x_grad_dwn	: first order difference downwind in x
	x_grad_up	: first order difference upwind in x
	y_grad 		: centered difference in y (unless boundary point)
	y_grad_dwn	: first order difference downwind in y
	y_grad_up	: first order difference upwind in y
	
	'''

	if n[0] == 0:
		#leftmost bdy
		x_grad_dwn = C[(n[0]+1,n[1])]-C[(n[0],n[1])]
		x_grad_up = x_grad_dwn
		x_grad = x_grad_dwn
	elif n[0] == x_n-1:
		#rightmost bdy
		x_grad_dwn = C[(n[0],n[1])]-C[(n[0]-1,n[1])]
		x_grad_up = x_grad_dwn
		x_grad = x_grad_dwn
	else:
		#interior point
		x_grad_dwn = C[(n[0],n[1])]-C[(n[0]-1,n[1])]
		x_grad_up = C[(n[0]+1,n[1])]-C[(n[0],n[1])]
		x_grad = (C[(n[0]+1,n[1])]-C[(n[0]-1,n[1])])/2

	if n[1] == 0:
		#downmost bdy
		y_grad_dwn = (C[(n[0],n[1]+1)]-C[(n[0],n[1])])
		y_grad_up = y_grad_dwn
		y_grad = y_grad_dwn
	elif n[1] == y_n-1:
		#upmost bdy
		y_grad_dwn = (C[(n[0],n[1])]-C[(n[0],n[1]-1)])
		y_grad_up = y_grad_dwn
		y_grad = y_grad_dwn
	else:
		#interior point
		y_grad_dwn = (C[(n[0],n[1])]-C[(n[0],n[1]-1)])
		y_grad_up = (C[(n[0],n[1]+1)]-C[(n[0],n[1])])
		y_grad = (C[(n[0],n[1]+1)]-C[(n[0],n[1]-1)])/2

	return x_grad,x_grad_dwn,x_grad_up,y_grad,y_grad_dwn, y_grad_up

def chi_grad_det(n,xn,yn,C_gradx,C_grady):

	'''
	Determines where to sample for up or downwind from the point n based on x- and y- gradients of C

	inputs:
	
	n 			: point where we are estimating the gradient
	x_n 		: max x-value
	y_n 		: max y-value
	C_gradx 	: Estimate of C gradient in x
	C_grady 	: Estimate of C gradient in y

	outputs:

	n_grad_x_up 	: point to sample for x upwind
	n_grad_x_dwn	: point to sample for x downwind
	n_grad_y_up 	: point to sample for y upwind
	n_grad_y_dwn	: point to sample for y downwind
	
	'''

	if C_gradx >= 0:
		if n[0] != xn-1:
			n_grad_x_up = (n[0]+1,n[1])
			n_grad_x_dwn = None
		else:
			n_grad_x_up = n
			n_grad_x_dwn = None
	else:
		if n[0] != 0:
			n_grad_x_up = None
			n_grad_x_dwn = (n[0]-1,n[1])
		else:
			n_grad_x_up = None
			n_grad_x_dwn = n

	if C_grady >= 0:
		if n[1] != yn-1:
			n_grad_y_up = (n[0],n[1]+1)
			n_grad_y_dwn = None
		else:
			n_grad_y_up = n
			n_grad_y_dwn = None
	else:
		if n[1] != 0:
			n_grad_y_up = None
			n_grad_y_dwn = (n[0],n[1]-1)
		else:
			n_grad_y_up = None
			n_grad_y_dwn = n

	return n_grad_x_up, n_grad_x_dwn, n_grad_y_up, n_grad_y_dwn

def prob_branch(C):

	'''
	Provides probability of a sprout branching based on the local chemoattractant gradient

	inputs:
	
	C 	: Chemoattractant density

	outputs:

	prob : probability of branching
	
	'''
	
	return 1.0
	'''if C < .1:#0.25:
					return 0
				elif C < .25:#.45:
					return 0.2
				elif C < .4:#0.6:
					return .3
				elif C < .5:#0.68:
					return 0.4
				else:
					return 1.0'''


def radial_persistence(title,dir_orient):

	'''
	Computes betti0 and 1 of perseus file located at title+"_"+dir using cubical complex.

	inputs:

	title : name of perseus file
	dir   : 'left', 'right', 'top', 'bottom'

	outputs: 

	count0 :  Betti0 (connected componenets topology)
	count1 :  Betti1 (loops topology)

	'''

	#create cubical complex
	cc = gd.CubicalComplex(perseus_file=title + '_' + dir_orient)
	
	#compute betti0, 1
	diag = cc.persistence(
	        homology_coeff_field=2, min_persistence=0)

	#initialize betti0, betti1
	count0 = 0
	count1 = 0

	# Add up betti 0, 1
	for d in diag:
	    #print(d)
	    if d[0] == 0:
	        count0+=1
	    elif d[0] == 1:
	    	count1+=1

	return count0, count1

def image_topology(N):

	###
	### Currently not working and I am diagnosing why
	###

	xm,ym = N.shape

	st = gd.SimplexTree()

	## look for vertical neighbors
	vert_neighbors = np.logical_and(N[:-1,:]==1,N[1:,:]==1)
	a = np.where(vert_neighbors)
	a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
	locs = a[:,0] + xm*a[:,1]
	for j in locs:
		st.insert([j,j+1],filtration = 0)

	## look for horizontal neighbors
	horiz_neighbors = np.logical_and(N[:,:-1]==1,N[:,1:]==1)
	a = np.where(horiz_neighbors)
	a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
	locs = a[:,0] + xm*a[:,1]
	for j in locs:
		st.insert([j,j+xm],filtration = 0)

	
	#look for diagonal neighbors (top left to bottom right)
	diag_neighbors = np.logical_and(N[:-1,:-1]==1,N[1:,1:]==1)
	a = np.where(diag_neighbors)
	a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
	locs = a[:,0] + xm*a[:,1]
	for j in locs:
		st.insert([j,j+xm+1],filtration = 0)
	

	#look for diagonal neighbors (bottom left to top right)
	diag_neighbors = np.logical_and(N[1:,:-1]==1,N[:-1,1:]==1)
	a = np.where(diag_neighbors)
	a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
	locs = a[:,0] + xm*a[:,1]
	for j in locs:
		st.insert([j+1,j+xm],filtration = 0)

	st.set_dimension(2)

	###include 2-simplices (looking for four different types of corners)

	for i in np.arange(xm-1):
		for j in np.arange(ym-1):


			#top left corner:
			if N[i,j]==1 and N[i+1,j]==1 and N[i,j+1]==1:
				st.insert([xm*i + j,xm*(i+1) + j , xm*i +j+1],filtration = 0)
			
			#top right corner
			if N[i,j]==1 and N[i+1,j]==1 and N[i+1,j+1]==1:
				st.insert([i*xm + j, (i+1)*xm+j, (i+1)*xm  + j+1],filtration = 0)

			#bottom left corner
			if N[i,j]==1 and N[i,j+1]==1 and N[i+1,j+1]==1:
				st.insert([xm*i + j, i*xm + j+1, xm*(i+1) + j+1],filtration = 0)

			#bottom right corner
			if N[i+1,j+1]==1 and N[i+1,j]==1 and N[i,j+1]==1:
				st.insert([xm*(i+1) + j + 1, xm*(i+1) + j, i*xm + j + 1],filtration = 0)

	diag = st.persistence(homology_coeff_field=2)

	return diag

def param_sweep(N,X,filename=None,iter_num = 50,plane_dir='less'):
	
	'''
	param_sweep

	inputs:
	N : Binary image
	iter_num : number of flooding events to compute

	output
	diag : Birth and death times for all topological features
	'''

	if N.shape != X.shape:
		raise Exception("Shape of N and X must the equal")

	xm,ym = N.shape

	r_range = np.linspace(0,np.max(X),iter_num)

	if plane_dir == 'greater':
		r_range = r_range[::-1]

	st = gd.SimplexTree()

	for k,rr in enumerate(r_range):

		#find nonzero pixels in N that are to the correct direction of the plane
		if plane_dir == 'less':
			N_update = np.logical_and(N,X<=rr)
		elif plane_dir == 'greater':
			N_update = np.logical_and(N,X>=rr)
		else:
			raise Exception("N_update must be \'less\' or \'greater\'")

		## look for vertical neighbors
		cell_loc = N_update==1
		a = np.where(cell_loc)
		a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
		locs = a[:,0] + xm*a[:,1]
		#locs = xm*a[:,0] + a[:,1]
		for j in locs:
			st.insert([j],filtration = k)

		## look for vertical neighbors
		vert_neighbors = np.logical_and(N_update[:-1,:]==1,N_update[1:,:]==1)
		a = np.where(vert_neighbors)
		a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
		locs = a[:,0] + xm*a[:,1]
		#locs = xm*a[:,0] + a[:,1]
		for j in locs:
			st.insert([j,j+1],filtration = k)

		## look for horizontal neighbors
		horiz_neighbors = np.logical_and(N_update[:,:-1]==1,N_update[:,1:]==1)
		a = np.where(horiz_neighbors)
		a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
		locs = a[:,0] + xm*a[:,1]
		
		for j in locs:
			st.insert([j,j+xm],filtration = k)


		#look for diagonal neighbors (top left to bottom right)
		diag_neighbors = np.logical_and(N_update[:-1,:-1]==1,N_update[1:,1:]==1)
		a = np.where(diag_neighbors)
		a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
		locs = a[:,0] + xm*a[:,1]
		#locs = xm*a[:,0] + a[:,1]
		for j in locs:
			st.insert([j,j+xm+1],filtration = k)
		

		#look for diagonal neighbors (bottom left to top right)
		diag_neighbors = np.logical_and(N_update[1:,:-1]==1,N_update[:-1,1:]==1)
		a = np.where(diag_neighbors)
		a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
		locs = a[:,0] + xm*a[:,1]
		
		for j in locs:
			st.insert([j+1,j+xm],filtration = k)

		st.set_dimension(2)

		###include 2-simplices (looking for four different types of corners)

		for j in np.arange(ym-1):
			for i in np.arange(xm-1):

				
				#top left corner:
				if N_update[i,j]==1 and N_update[i+1,j]==1 and N_update[i,j+1]==1:
					st.insert([i + xm*j,(i+1) + xm*j , i + xm*(j+1)],filtration = k)
				
				#top right corner
				if N_update[i,j]==1 and N_update[i+1,j]==1 and N_update[i+1,j+1]==1:
					st.insert([i + j*xm, (i+1)+j*xm, (i+1)  + (j+1)*xm],filtration = k)

				#bottom left corner
				if N_update[i,j]==1 and N_update[i,j+1]==1 and N_update[i+1,j+1]==1:
					st.insert([i + j*xm, i + (j+1)*xm, (i+1) + (j+1)*xm],filtration = k)

				#bottom right corner
				if N_update[i+1,j+1]==1 and N_update[i+1,j]==1 and N_update[i,j+1]==1:
					st.insert([(i+1) + (j + 1)*xm, (i+1) + j*xm, i + (j + 1)*xm],filtration = k)

	
	diag = st.persistence()

	if filename is not None:

		data = {}
		data['BD'] = diag
		np.save(filename,data)


	return diag


def level_set_flooding(N,filename=None,iter_num = 50):
	
	'''
	level_set_flooding

	inputs:
	N : Binary image
	iter_num : number of flooding events to compute

	output
	diag : Birth and death times for all topological features
	'''


	xm,ym = N.shape

	st = gd.SimplexTree()

	kernel = np.ones((3,3),np.uint8)

	N = cv2.dilate(N,kernel,iterations=1)

	for iterate in np.arange(iter_num):

		if iterate != 0:
		    N = cv2.dilate(N,kernel,iterations=1)

		## look for vertical neighbors
		vert_neighbors = np.logical_and(N[:-1,:]==1,N[1:,:]==1)
		a = np.where(vert_neighbors)
		a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
		locs = a[:,0] + xm*a[:,1]
		for j in locs:
			st.insert([j,j+1],filtration = iterate)

		## look for horizontal neighbors
		horiz_neighbors = np.logical_and(N[:,:-1]==1,N[:,1:]==1)
		a = np.where(horiz_neighbors)
		a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
		locs = a[:,0] + xm*a[:,1]
		for j in locs:
			st.insert([j,j+xm],filtration = iterate)

		#look for diagonal neighbors (top left to bottom right)
		diag_neighbors = np.logical_and(N[:-1,:-1]==1,N[1:,1:]==1)
		a = np.where(diag_neighbors)
		a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
		locs = a[:,0] + xm*a[:,1]
		for j in locs:
			st.insert([j,j+xm+1],filtration = iterate)
		

		#look for diagonal neighbors (bottom left to top right)
		diag_neighbors = np.logical_and(N[1:,:-1]==1,N[:-1,1:]==1)
		a = np.where(diag_neighbors)
		a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
		locs = a[:,0] + xm*a[:,1]
		for j in locs:
							st.insert([j+1,j+xm],filtration = iterate)

		st.set_dimension(2)

		'''## look for squares of neighbors (vert,horiz, AND diag neighbors)
								diag_neighbors = np.logical_and(N[:-1,:-1]==1,N[1:,1:]==1)
								diag_neighbors = np.logical_and(diag_neighbors,vert_neighbors[:,:-1])
								diag_neighbors = np.logical_and(diag_neighbors,horiz_neighbors[:-1,:])
								a = np.where(diag_neighbors)
								a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
								locs = a[:,0] + xm*a[:,1]
								for j in locs:
									st.insert([j,j+1,j+xm,j+xm+1],filtration = iterate)'''

		###include 2-simplices (looking for four different types of corners)

		###include 2-simplices (looking for four different types of corners)

		for j in np.arange(ym-1):
			for i in np.arange(xm-1):

				#### indices are flipped incorrectly.
				#top left corner:
				if N[i,j]==1 and N[i+1,j]==1 and N[i,j+1]==1:
					st.insert([i + xm*j,(i+1) + xm*j , i + xm*(j+1)],filtration = iterate)
				
				#top right corner
				if N[i,j]==1 and N[i+1,j]==1 and N[i+1,j+1]==1:
					st.insert([i + j*xm, (i+1)+j*xm, (i+1)  + (j+1)*xm],filtration = iterate)

				#bottom left corner
				if N[i,j]==1 and N[i,j+1]==1 and N[i+1,j+1]==1:
					st.insert([i + j*xm, i + (j+1)*xm, (i+1) + (j+1)*xm],filtration = iterate)

				#bottom right corner
				if N[i+1,j+1]==1 and N[i+1,j]==1 and N[i,j+1]==1:
					st.insert([(i+1) + (j + 1)*xm, (i+1) + j*xm, i + (j + 1)*xm],filtration = iterate)

	
	diag = st.persistence()
	if filename is not None:

		data = {}
		data['BD'] = diag
		np.save(filename,data)


	return diag


def weight_fun_ramp(x,**options):

	'''
	Weight function for persistence images

	inputs 

	x : function input
	b : max x value

	outputs 

	y: function output
	'''

	b = options.get("b")

	y = np.zeros(x.shape)

	samp = np.where(x<=0)[0]
	y[samp] = np.zeros(samp.shape)

	samp = np.where(np.logical_and(x>0,x<b))[0]
	y[samp] = x[samp]/b

	samp = np.where(x>=b)[0]
	y[samp] = np.ones(samp.shape)

	return y


def weight_fun_1(x,**options):

	'''
	Weight function of 1's for persistence images

	inputs 

	x: function input

	outputs 

	y: function output
	'''

	y = np.ones(x.shape)

	return y



def betti_curve(diag=None,filename=None,filename_save=None,r0=0,r1=1,rN=40):

	'''
	betti_curve construction

	inputs

	diag :          Input Birth-death interval list. If none, then this will be loaded in
	filename: 		Where Birth-death interval list is stored
	filename_save	Where to save persistence image

	output

	IP : 			Persistence Image
	'''


	if diag is None:
		
		if filename is None:
			raise Exception("Either interval data or filename for one must be provided")
		mat = np.load(filename + '.npy',allow_pickle=True, encoding='latin1').item()

		diag = mat['BD']

	r_range = np.linspace(r0,r1,rN)

	b0 = np.zeros(r_range.shape)
	b1 = np.zeros(r_range.shape)

	for i,r in enumerate(r_range):
		for dd in diag:
			if r >= dd[1][0] and r < dd[1][1]:
				if dd[0] == 0:
					b0[i] += 1
				elif dd[0] == 1:
					b1[i] += 1

	if filename_save is not None:
		data = {}
		data['b0'] = b0
		data['b1'] = b1
		data['r'] = r_range
		np.save(filename_save,data)

	return b0,b1,r_range



def Persist_im(diag=None,filename=None,filename_save=None,inf_val=25,sigma=1e-1,weight_fun=weight_fun_ramp):

	'''
	create persistence image

	inputs

	diag :          Input Birth-death interval list. If none, then this will be loaded in
	filename: 		Where Birth-death interval list is stored
	filename_save	Where to save persistence image

	output

	IP : 			Persistence Image
	'''
    
	if diag is None:
		
		if filename is None:
			raise Exception("Either interval data or filename for one must be provided")
		mat = np.load(filename + '.npy',allow_pickle=True, encoding='latin1').item()

		diag = mat['BD']

	#resolution of final persistance image will be res**2
	res = 50	
	
	### Convert to non-diagonal form
	BD_list = [np.zeros((1,2)),np.zeros((1,2))]

	b0 = 0
	b1 = 0
	for dd in diag:
		if dd[0] == 0:

			if b0 == 0:
				BD_list[0][0,:] = dd[1]
			else:
				BD_list[0] = np.vstack((BD_list[0],dd[1]))
			
			b0 += 1

		elif dd[0] == 1:

			if b1 == 0:
				BD_list[1][0,:] = dd[1]
			else:
				BD_list[1] = np.vstack((BD_list[1],dd[1]))

			b1 += 1

	Ip_ones = [np.zeros((res,res)),np.zeros((res,res))]
	Ip_ramp = [np.zeros((res,res)),np.zeros((res,res))]

	for i,BD in enumerate(BD_list):

		BD[np.isinf(BD)] = inf_val
		BD_adjust = np.hstack([BD[:,0][:,np.newaxis],(BD[:,1] - BD[:,0])[:,np.newaxis]])

		width,height = np.max(BD_adjust,axis=0)
		length = inf_val#np.max((width,height))
		U = BD_adjust.shape[0]

		x = np.linspace(0,length,res+1)
		y = np.linspace(0,length,res+1)

		X,Y = np.meshgrid(x,y)

		shape = X.shape

		weights_ones = weight_fun_1(BD_adjust[:,1],b=height)
		weights_ramp = weight_fun_ramp(BD_adjust[:,1],b=height)

		for j,bd in enumerate(BD_adjust):

			Ip_tmp = np.zeros((res+1,res+1))
			for k,xx in enumerate(x):
				for l,yy in enumerate(y):
					Ip_tmp[k,l] = multivariate_normal.cdf(np.hstack((xx,yy)),
															mean=bd,
															cov=sigma)
			
			#Use summed area table (coordinates reverse of those described in wikipedia)
			Ip_ones[i] +=  weights_ones[j]*(Ip_tmp[1:,1:] + Ip_tmp[:-1,:-1] - Ip_tmp[1:,:-1] - Ip_tmp[:-1,1:])
			Ip_ramp[i] +=  weights_ramp[j]*(Ip_tmp[1:,1:] + Ip_tmp[:-1,:-1] - Ip_tmp[1:,:-1] - Ip_tmp[:-1,1:])


	if filename_save is not None:
		data = {}
		data['Ip'] = Ip_ones
		np.save(filename_save[0],data)

		data = {}
		data['Ip'] = Ip_ramp
		np.save(filename_save[1],data)

	return Ip_ones,Ip_ramp


def run_sweep_TDA(N,X,Y,plane_diff=0):

	'''
	compute and save betti 0 and 1 curves for left to right topology
	'''
	
	#values of the sliding plane will take on
	r_range = np.linspace(0,1,40)

	features = []
	features_desc = []

	dirs = ['left']

	# loop through r values
	for dd in dirs:

		#initialize betti0, betti 1 curves
		b0_pers = np.zeros(len(r_range))
		b1_pers = np.zeros(len(r_range))


		for j,rr in enumerate(r_range):


			#make perseus file of pixels
			make_image_two_planes(N,X,Y,"angio",r_up=rr,dir=dd,r_low=rr-plane_diff)

			#analyze cubical complex topology of the image
			b0l, b1l = radial_persistence("angio",dir_orient=dd)

			#record b0,b1
			b0_pers[j] = b0l
			b1_pers[j] = b1l

		#save to file
		
		features.append(copy.deepcopy(b0_pers))
		features.append(copy.deepcopy(b1_pers))

		features_desc.append(copy.deepcopy('b0_'+dd))
		features_desc.append(copy.deepcopy('b1_'+dd))


	return features, features_desc


def make_image_two_planes(N,X,Y,title,r_up,dir="left",r_low=0):
		
	'''
	Create perseus file from vessel network. This is currently using two sliding plane,
	so pixels to the left of r_up and the right of r_low are recorded

	description of perseus files provided at the bottom of
	http://gudhi.gforge.inria.fr/python/latest/fileformats.html

	inputs:

	title 	 : 	title of perseus file (will be saved)
	r 	     : 	x pixel location of plane
	'''

	if r_low > r_up:
		raise Exception("r_up must exceed r_low")
		

	#initialize file
	file = open(title + "_" + dir,"w")


	m,n = N.shape
	file.write("2\n")
	file.write(str(m)+"\n")
	file.write(str(n)+"\n")
	for i in np.arange(m-1,-1,-1):
		for j in np.arange(n):

			#only consider points to left of r
			if dir == "left":

				if r_low <= X[i,j] and X[i,j] <= r_up:
					file.write(str(1-N[i,j])+"\n")
				else:
					file.write("1\n")


			'''#only consider points to right of r
			elif dir == "right":

				if r_down self.X[i,j] >= r_up:
					file.write(str(1-self.N[i,j])+"\n")
				else:
					file.write("1\n")

			#only consider points below r
			elif dir == "bottom":

				if self.Y[i,j] <= r_up:
					file.write(str(1-self.N[i,j])+"\n")
				else:
					file.write("1\n")

			#only consider points above r
			elif dir == "top":

				if self.Y[i,j] >= r_up:
					file.write(str(1-self.N[i,j])+"\n")
				else:
					file.write("1\n")'''

	file.close()

class angio_abm :

	def __init__(self,
				 IC = 'linear',
				 rho = 0.34,
				 t_final = 4.0,
				 chi = 0.38,
				 chemo_rate = 'const',
				 psi = 0.5):

		##parameters
		self.D = .00035
		self.alpha = 0.6
		self.chi = chi
		self.rho = rho#0.034 ##0.34
		self.beta = 0.05 ### 1?
		self.gamma = 0.1
		self.eta = 0.1
		self.psi = psi

		self.nb_const = 2.5

		self.chemo_rate = chemo_rate

		##grids
		
		self.eps1 = 0.45
		self.eps2 = 0.45
		self.k    = 0.75
		self.nu   = (np.sqrt(5) - 0.1)/(np.sqrt(5)-1)

		self.xn = 201
		self.yn = 201
		self.x = np.linspace(0,1,self.xn)
		self.y = np.linspace(0,1,self.yn)
		self.IC = IC
		self.dx = self.x[1] - self.x[0]
		self.dy = self.y[1] - self.y[0]


		self.dt = 0.01
		self.t_final = t_final#np.int(4.0//self.dt)
		self.time_grid = np.arange(0,self.t_final,self.dt)

		self.write_folder = 'results/'

		#self.file_name = 'IC_'+self.IC + '_rho_'+str(round(self.rho,2))+'_chi_'+str(round(self.chi,2))
		self.file_name = 'IC_'+self.IC + '_rho_'+str(round(self.rho,2))+'_chi_'+str(round(self.chi,2)) + '_psi_' + str(round(self.psi,2))
		
		if os.path.isdir(self.write_folder) == False:
			os.mkdir(self.write_folder)


		#initialize sprouts

		self.cell_locs = [[0,.1],[0,.2],[0,.3],[0,.4],[0,.5],[0,.6],[0,.7],[0,.8],[0,.9]]#[0,.17],[0,.3],[0,.5],[0,.65],[0,.84]]		
		self.sprouts = []
		self.sprout_ages = []

		self.branches = 0

		## time-dependent things to save
		self.sprouts_time = []
		self.branches_time = []
		self.active_tips = []

	def IC_generate(self):


		'''
		Sets the initial conditions for n , C, and F. 

		'''

		self.Y,self.X = np.meshgrid(self.y,self.x)

		#TAF
		if self.IC == 'tumor':
			# Equationa 10 & 11 in Anderson-Chaplain
			r = np.sqrt((self.X-1)**2 + (self.Y-0.5)**2)
			self.C = (self.nu - r)**2/(self.nu-0.1) / 1.68
			self.C[r<=0.1] = 1

		elif self.IC == 'linear':
			# Equation 12 in Anderson-Chaplain

			self.C = np.exp(-((1-self.X)**2)/self.eps1)

		#fibronectin
		# Equation 13 in Anderson-Chaplain
		self.F = self.k*np.exp(-self.X**2/self.eps2)

		#endo_loc
		self.N = np.zeros(self.X.shape)

	def chemotaxis_rate(self,C):

		'''
		Sets the chemotaxis rate (as a function of C) for the model

		outputs:

		chi 	: Chemotaxis rate (const is constant , hill returns hill function).

		'''

		if self.chemo_rate == 'hill':
			#### from Chap-Anderson paper.
			return self.chi/(1+self.alpha*self.C[n])
		elif self.chemo_rate == 'const':
			return self.chi

	def sprout_initialize(self):

		'''
		Set the initial sprout locations for the model

		'''
		
		for c in self.cell_locs:
			#self.new_sprout([tuple((np.where(self.x==0)[0][0],np.argmin(np.abs(self.y-c))))])
			self.new_sprout([tuple((np.argmin(np.abs(self.x-c[0])),np.argmin(np.abs(self.y-c[1]))))])

	def record_bio_data(self):

		'''
		Record the number of sprouts, branches, and active tip cells in the model over time

		'''
		
		self.sprouts_time.append(len(self.sprouts))
		self.branches_time.append(self.branches)
		self.active_tips.append(sum(np.array(self.sprout_ages) != -1))
		

	def save_bio_data(self,num):

		'''
		Save the sprouts, branches, tips, time, and overall network to memory

		'''
		

		data = {}
		data['sprouts'] = self.sprouts_time
		data['branches'] = self.branches_time
		data['active_tips'] = self.active_tips
		data['t'] = self.time_grid
		data['N'] = self.N

		np.save(self.write_folder  + 'angio_bio_data_' + self.file_name+'_real_'+str(num),data)


	def move_sprouts(self):

		'''
		Update the tip cell locations and overall network based on tip cell movement.

		'''
		

		for i,nl in enumerate(self.sprouts):

			#sprout no longer moving if it anastamosed (age = -1 for anasotomosed sprouts)
			if self.sprout_ages[i] == -1:
				continue

			# get current tip cell
			n = nl[-1]

			#sample local gradients
			C_gradx,C_gradx_dwn,C_gradx_up,C_grady,C_grady_dwn,C_grady_up = grad_est(n,self.xn,self.yn,self.C)
			F_gradx,F_gradx_dwn,F_gradx_up,F_grady,F_grady_dwn,F_grady_up = grad_est(n,self.xn,self.yn,self.F)

			#determine indices of up/down wind based on gradients.
			n_x_up, n_x_dwn, n_y_up, n_y_dwn = chi_grad_det(n,self.xn,self.yn,C_gradx_up,C_grady_up) 
			

			#### Move tip cells: P0 is the probability a cell stays put, P1-4 are the probabilities
			#### of moving right, left, up, and down, respectively. 
			
			#beginning by defining these probabilities from just diffusion
			#start with just diffusion
			P0     = 1.0 - 4.0*self.dt/(self.dx**2)*self.D

			#move right,left,up,down
			P1, P2, P3, P4     = self.dt/(self.dx**2)*self.D,self.dt/(self.dx**2)*self.D,self.dt/(self.dx**2)*self.D,self.dt/(self.dx**2)*self.D

			### now incorporate chemotaxis
			# increasing chemical gradient -- sample downwind
			if C_gradx > 0 :
				P0 += -self.dt/(self.dx**2)*(self.chemotaxis_rate(self.C[n])*C_gradx_dwn)
				P1 +=  self.dt/(self.dx**2)*(self.chemotaxis_rate(self.C[n_x_up])*C_gradx_up)
			elif C_gradx < 0:
				# deccreasing chemical gradient -- sample upwind
				P0 += self.dt/(self.dx**2)*(self.chemotaxis_rate(self.C[n])*C_gradx_up)
				P2 += -self.dt/(self.dx**2)*(self.chemotaxis_rate(self.C[n_x_dwn])*C_gradx_dwn)
			#Do the same in the y-dimension
			if C_grady > 0:
				P0 += -self.dt/(self.dx**2)*(self.chemotaxis_rate(self.C[n])*C_grady_dwn)
				P3 +=  self.dt/(self.dx**2)*(self.chemotaxis_rate(self.C[n_y_up])*C_grady_up)
			elif C_grady < 0:
				# deccreasing chemical gradient -- sample upwind
				P0 += self.dt/(self.dx**2)*(self.chemotaxis_rate(self.C[n])*C_grady_up)
				P4 += -self.dt/(self.dx**2)*(self.chemotaxis_rate(self.C[n_y_dwn])*C_grady_dwn)
			

			###haptotaxis
			# increasing chemical gradient, then sample downwind
			if F_gradx > 0 :
				P0 += self.rho*self.dt/(self.dx**2)*(F_gradx_dwn)
				P2 += -self.rho*self.dt/(self.dx**2)*(F_gradx_up)
			elif F_gradx < 0:
				# decreasing chemical gradient, then sample upwind
				P0 += -self.rho*self.dt/(self.dx**2)*(F_gradx_up)
				P1 += self.rho*self.dt/(self.dx**2)*(F_gradx_dwn)
			
			#do the same for y
			if F_grady > 0:
				P0 += self.rho*self.dt/(self.dx**2)*(F_grady_dwn)
				P4 += -self.rho*self.dt/(self.dx**2)*(F_grady_up)
			
			elif F_grady < 0:
				P0 += -self.rho*self.dt/(self.dx**2)*(F_grady_up)
				P3 += self.rho*self.dt/(self.dx**2)*(F_grady_dwn)
			

			#now we have our final probabilities
			total = P0 + P1 + P2 + P3 + P4

			#determine random number
			p = np.random.uniform(low=0,high = total)

			if p < P0:
				#stay put
				nl.append(n)
				moved = False
			elif  p < P1+P0:
				#move right
				if n[0] <= len(self.x)-2:
					nl.append((n[0]+1,n[1]))
					moved = True
				else:
					nl.append(n)
					moved = False
			elif p < P2+P1+P0:
				#move left
				if n[0] > 0: 
					nl.append((n[0]-1,n[1]))
					moved = True
				else:
					nl.append(n)
					moved = False
			elif p < P3+P2+P1+P0:
				#move up
				if n[1] < len(self.y)-1:
					nl.append((n[0],n[1]+1))    
					moved = True
				else:
					nl.append(n)
					moved = False
			elif p <= P4+P3+P2+P1+P0: 
				#move down
				if n[1] > 0:  
					nl.append((n[0],n[1]-1))
					moved = True
				else:
					nl.append(n)
					moved = False
			else:
				moved = False
			
			#anastomsis occurs if vessel moves into occupied space
			if self.N[nl[-1]] == 1 and moved==True:
				#list through other sprouts
				for j,sprout in enumerate(self.sprouts):
					#don't search same sprout
					if i != j:	
						#cell no longer active
						if nl[-1] in sprout and self.sprout_ages[j]!= -1:
							self.sprout_ages[i] = -1


			### Update sprout network
			if self.N[nl[-1]] != 1:
				self.N[nl[-1]] = 1


	def update_grids(self):

		'''
		Update F and C using the equations from the appendix of Anderson-Chaplain
		'''
		
		for i,nl in enumerate(self.sprouts):
			n = nl[-1]

			self.F[n] = self.F[n]*(1-self.dt*self.gamma*1) + self.dt*self.beta*1
			self.C[n] = self.C[n]*(1-self.dt*self.eta*1)

	def branch(self):


		'''
		Determine if a tip cell should branch. If so, determine where daughter cell is placed

		Rules for tip sprouting are outlined in Section 4.1 of Anderson-Chaplain
		'''
		

		for i,nl in enumerate(self.sprouts):

			#sprout no longer branching if it anastamosed (sprout_ages[i] = -1)
			if self.sprout_ages[i] == -1:
				continue

			#get i-th tip cell.
			n = nl[-1]

			branch = False

			#Rule 1: branch when age over psi
			if self.sprout_ages[i] > self.psi:

				#which neighboring spots are available?
				# sample over the 3x3 grid [x_rang,y_rang] including n.
				if n[0]==0:
					x_rang = np.arange(n[0],n[0]+2)
				elif n[0] == self.xn-1:
					x_rang = np.arange(n[0]-1,n[0]+1)
				else:
					x_rang = np.arange(n[0]-1,n[0]+2)
				
				if n[1]==0:
					y_rang = np.arange(n[1],n[1]+2)
				elif n[1] == self.yn-1:
					y_rang = np.arange(n[1]-1,n[1]+1)
				else:
					y_rang = np.arange(n[1]-1,n[1]+2)

				xx,yy = np.meshgrid(x_rang,y_rang)
				xx = xx.reshape(-1)
				yy = yy.reshape(-1)

				#number of sprout cells in surrounding 3x3 grid
				avail = np.where(self.N[xx,yy]!=0)[0]

				
				#Rule 2: is there space to branch?
				if len(avail) > 0 :


					#Rule 3: is endothelial density above some threshold?
					if len(avail/9) > self.nb_const/self.C[n]:


						#prob_branching based on C

						pb = prob_branch(self.C[n])

						#branch with prob pb
						if np.random.uniform() < pb: 

							#select one of the available spaces for new sprout
							new_ind = np.random.permutation(len(avail))[0]

							#new sprout
							self.new_sprout([(xx[new_ind],yy[new_ind])])
							branch = True
							#set age back to zero
							self.sprout_ages[i] = 0
							#increase branches
							self.branches += 1

			    
			    
			#increase age if we never branched
			if branch == False:
				self.sprout_ages[i] += self.dt


			
	def new_sprout(self,loc):

		'''
		Create new sprout

		inputs :

		loc 	: list of coordinates of starting point of new branch	 
		'''
		

		self.sprouts.append(loc)
		self.sprout_ages.append(0)
		self.N[loc[0]] = 1
		

	def F_contour(self,title=None):

		'''
		make contour of F concentration

		inputs:
		title: title of figure to save (if None, then will not save)
		'''
		fontsize = 18
		        
		fig = plt.figure()
		ax = fig.add_subplot()
		cf = ax.contourf(self.X,self.Y,self.F,levels = np.linspace(0,np.max(self.F),11))
		fig.colorbar(cf)
		ax.set_xlabel('Space ($x$)',fontsize=fontsize)
		ax.set_ylabel('Space ($y$)',fontsize=fontsize)
		ax.set_title('Fibronectin, ' + self.IC + " profile",fontsize=fontsize)
		if title == None:
			plt.show()
		else:
			plt.savefig('figures/' + title+'.pdf', format='pdf')

	def C_contour(self,title=None):

		'''
		make contour of C concentration

		inputs:
		title: title of figure to save (if None, then will not save)
		'''
		fontsize = 18
		        
		
		fig = plt.figure()
		ax = fig.add_subplot()
		cf = ax.contourf(self.X,self.Y,self.C,levels = np.linspace(0,np.max(self.C),11))
		fig.colorbar(cf)
		ax.set_xlabel('Space ($x$)',fontsize=fontsize)
		ax.set_ylabel('Space ($y$)',fontsize=fontsize)
		ax.set_title('TAF, ' + self.IC + " profile",fontsize=fontsize)
		if title == None:
			plt.show()
		else:
			plt.savefig('figures/' + title+'.pdf', format='pdf')


	
	def plot_bio_summaries(self,t):


		'''
		makes plot of different branches (marked by different colors) and tip cell locations
		(marked) by different dots.
		'''
		

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_title("Biological Summaries, sprouts = " +str(self.sprouts_time[-1])+ ", tips = " + str(self.active_tips[-1]))
		
		for i,n in enumerate(self.sprouts):
			idx2=tuple(np.array(n).T)
			plt.plot(self.X[idx2],self.Y[idx2],zorder=0)

			if self.sprout_ages[i] != -1:
				plt.scatter(self.X[n[-1]],self.Y[n[-1]],c='blue',s=20,zorder=1)


		ax.set_xlim((0,1))
		ax.set_ylim((0,1))
		
		plt.savefig("ChapAndSim_"+str(t)+".png",dvips=500)


	def plot_sprouts(self,t):

		'''
		Save image of vessel network 

		'''
		

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_title(str(t))
		ax.contourf(self.X,self.Y,self.C)
		
		for n in self.sprouts:
			idx2=tuple(np.array(n).T)
			plt.plot(self.X[idx2],self.Y[idx2],'w')

		plt.savefig("ChapAndSim_"+t+".png",dvips=500)

	
	def save_LR_TDA(self,num):

		'''
		compute and save betti 0 and 1 curves for left to right topology
		'''
		
		#values of the sliding plane will take on
		r_range = np.linspace(0,1,40)

		#orientations
		dirs = ['left','right','top','bottom']

		data = {}
		data['r'] = r_range

		# loop through r values
		for dd in dirs:

			#initialize betti0, betti 1 curves
			b0_pers = np.zeros(len(r_range))
			b1_pers = np.zeros(len(r_range))


			for j,rr in enumerate(r_range):


				#make perseus file of pixels
				self.make_image("angio",r=rr,dir=dd)

				#analyze cubical complex topology of the image
				b0l, b1l = radial_persistence("angio",dir_orient=dd)

				#record b0,b1
				b0_pers[j] = b0l
				b1_pers[j] = b1l

			#save to file
			
			data['b0_'+dd] = copy.deepcopy(b0_pers)
			data['b1_'+dd] = copy.deepcopy(b1_pers)

			

		np.save(self.write_folder + 'angio_TDA_data_'+ self.file_name +'_real_'+str(num),data)
		
	def save_flooding_TDA(self,num):

		'''
		compute level-set flooding topology and corresponding persistence image
		'''

		#compute persistence diagram for simulation
		diag = level_set_flooding(self.N,filename = self.write_folder + 'angio_flood_bd_data_' +
			self.file_name + '_real_'+str(num),iter_num=25)

		#compute and save persistance image
		b0,b1,r = betti_curve(diag = diag, filename_save = self.write_folder + 'angio_flood_Betti_' +self.file_name + '_real_'+str(num),r0 = 0, r1 = 25, rN = 25)


		file_name_1 = self.write_folder + 'angio_flood_persim_ones_' + self.file_name + '_real_'+str(num)
		file_name_ramp = self.write_folder + 'angio_flood_persim_ramp_' + self.file_name + '_real_'+str(num)

		#compute and save persistance image
		Ip_ones,Ip_ramp = Persist_im(diag = diag, filename_save = [file_name_1,file_name_ramp],
							inf_val=25,sigma=1)



	def plane_sweeping_TDA(self,real):

		'''
		compute level-set flooding topology and corresponding persistence image
		'''

		orients = ['left','right']#,'top','bottom']

		for orient in orients:

			if orient == "left":
				plane_dir = 'less'
				indep_var = self.X
			elif orient == "right":
				plane_dir = 'greater'
				indep_var = self.X
			elif orient == "top":
				plane_dir = 'greater'
				indep_var = self.Y
			elif orient == "bottom":
				plane_dir = 'less'
				indep_var = self.Y



			#compute persistence diagram for simulation
			diag = param_sweep(self.N,indep_var,iter_num=51, plane_dir=plane_dir,
								filename=self.write_folder + 'angio_plane_'+orient+'_bd_data_' +
								self.file_name + '_real_'+str(real))


			#compute and save persistance image
			b0,b1,r = betti_curve(diag, r0 = 0, r1 = 51, rN = 25,
									filename_save=self.write_folder + 'angio_plane_'+orient+'_Betti_' +
									self.file_name + '_real_'+str(real))

			#compute and save persistance image
			file_name_1 = self.write_folder + 'angio_plane_'+orient+'_persim_ones_' + self.file_name + '_real_'+str(real)
			file_name_ramp = self.write_folder + 'angio_plane_'+orient+'_persim_ramp_' + self.file_name + '_real_'+str(real)

			Ip_ones,Ip_ramp = Persist_im(diag = diag,inf_val=50.0,sigma=(2.0**2),
				filename_save = [file_name_1,file_name_ramp])



	def gif_initialize(self):

		'''
		initialize gif
		'''

		self.images = []
		self.image_count = 0

	def gif_append(self,title):
		
		'''
		Make gif by appending images of current vessel network
		'''


		fig = plt.figure()
		plt.title(title)
		plt.contourf(self.X,self.Y,self.C)
		#for n in n_loc:
		#    ax.scatter(X[n],Y[n],c='k')
		for n in self.sprouts:
			idx2=tuple(np.array(n).T)
			plt.plot(self.X[idx2],self.Y[idx2],'w')
		filename = "figures/gif_"+str(self.image_count)+".png"
		self.image_count += 1
		plt.savefig(filename,dvips=500)
		self.images.append(imageio.imread(filename))

	def gif_finalize(self,title):

		'''
		Finalize gif by saving and removing figures.
		'''


		imageio.mimsave('figures/'+title+'.gif', self.images,fps = 3)
		
		gif_list = glob.glob('figures/gif_*')
		for g in gif_list:
			os.remove(g)

	def make_image(self,title,r,dir):
		
		'''
		Create perseus file from vessel network. This is currently using a sliding plane,
		so only pixels to the left of r are recorded

		description of perseus files provided at the bottom of
		http://gudhi.gforge.inria.fr/python/latest/fileformats.html

		inputs:

		title 	 : 	title of perseus file (will be saved)
		r 	     : 	x pixel location of plane
		'''

		#initialize file
		file = open(title + "_" + dir,"w")


		m,n = self.N.shape
		file.write("2\n")
		file.write(str(m)+"\n")
		file.write(str(n)+"\n")
		for i in np.arange(m-1,-1,-1):
		    for j in np.arange(n):
		    	
		    	#only consider points to left of r
		    	if dir == "left":

			    	if self.X[i,j] <= r:
			    		file.write(str(1-self.N[i,j])+"\n")
		    		else:
		    			file.write("1\n")


    			#only consider points to right of r
		    	elif dir == "right":

			    	if self.X[i,j] >= r:
			    		file.write(str(1-self.N[i,j])+"\n")
		    		else:
		    			file.write("1\n")

    			#only consider points below r
		    	elif dir == "bottom":

			    	if self.Y[i,j] <= r:
			    		file.write(str(1-self.N[i,j])+"\n")
		    		else:
		    			file.write("1\n")

    			#only consider points above r
		    	elif dir == "top":

			    	if self.Y[i,j] >= r:
			    		file.write(str(1-self.N[i,j])+"\n")
		    		else:
		    			file.write("1\n")

		file.close()