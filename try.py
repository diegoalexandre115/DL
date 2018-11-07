import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import mnist

def normbatch(batch):

	for b in range(batch.shape[0]):
			me = np.mean(batch[b],axis=(0,1,2))
			#print(me)
			std = np.std(batch[b],axis=(0,1,2))
			#print(std)
			batch[b] = (batch[b] - me)/std
	return batch

def batchsep(conj,bsize):
   
	conj2 = []
	conj = np.asarray(conj)
	for i in range(0,conj.shape[0],bsize):
	   
	   conj2.append(conj[i:i+bsize])
	#return(np.asarray(conj2))
	#print(conj2)   
	return normbatch(np.asarray(conj2, dtype=float))

def batchsep2(conj,bsize):
   
	conj2 = []
	conj = np.asarray(conj)
	for i in range(0,conj.shape[0],bsize):
	   
	   conj2.append(conj[i:i+bsize])
	#return(np.asarray(conj2))
	#print(conj2)   
	return np.asarray(conj2)                   



def relu(img):
	img = np.asarray(img)
	for i in range(len(img.flat)):
		img.flat[i] = max(0,img.flat[i])
	return img

def he(cLayer_size,size,pLayer_size):
	#parametros:
	#cLayer = tamanho da camada atual
	#pLayer = tamanho da camada anterior
	#w=np.random.randn(cLayer_size,size)*np.sqrt(2/pLayer_size)
	std_dev = np.sqrt(2.0 / (cLayer_size + size))
	return std_dev*np.random.randn(cLayer_size, size)
	#return w

def softmax(img):
	img = np.asarray(img)
	img = np.exp(img)/np.sum(np.exp(img))
	return img

def maxpool(img, size, band=0):
	#parametros 
	#img = imagem de entrada
	#size = tamanho da matriz de pool
	rec = []
	img = np.asarray(img)
	raio = size//2
	sh = len(img.shape)
	img2 = []
	k =0;
	l = 0;

	if(sh>2):
	 
		img2 = np.zeros((img.shape[0],int(img.shape[1]//2),int(img.shape[2]//2)))
		for i in range(raio,img.shape[1]-raio+1,2):
			for j in range(raio,img.shape[2]-raio+1,2):
				for s in range(img.shape[0]):
					rec = img[s,i-raio : i+raio,j-raio : j+raio]
					img2[s,k, l] = np.amax(rec)
				l+=1
			k+=1
			l =0    
	else:
	  
		img2 = np.zeros((int(img.shape[0]//2),int(img.shape[1]//2)))
		for i in range(raio,img.shape[0]-raio+1,2):
			for j in range(raio,img.shape[1]-raio+1,2):
				rec = img[i-raio : i+raio,j-raio : j+raio]
				img2[k, l] = np.amax(rec)
				l+=1
			k+=1
			l =0    

	return img2            


def padd(img, raio):
	sh = len(img.shape)
	if (sh > 2):
		for i in range(raio):
			img =  np.insert(img,0,0,axis = 3)
			img = np .insert(img,img.shape[3],0,axis =3)
			img =  np.insert(img,0,0,axis = 2)
			img =  np.insert(img,img.shape[2],0,axis = 2)
	else:
		for i in range(raio):
			img =  np.insert(img,0,0,axis = 0)
			img =  np.insert(img,img.shape[0],0,axis = 0)
			img =  np.insert(img,0,0,axis = 1)
			img =  np.insert(img,img.shape[1],0,axis = 1)
	return img


def conv(img,filt):
	filt = np.asarray(filt)
	sfilt = filt.shape[1]
	#print(filt.shape)
	simg = img.shape[1]
	raio = sfilt//2
	sh = len(filt.shape)
	rec = []
	l = 0
	if(len(img.shape) > 2):
		img2 = np.zeros((int(((img.shape[1]-sfilt))+1),int(((img.shape[2]-sfilt))+1)))#,dtype = object)
	else:	
		img2 = np.zeros((int(((img.shape[0]-sfilt))+1),int(((img.shape[1]-sfilt))+1)))#,dtype = object)
	if(sh >2 ):
		if(len(img.shape) > 2):
			for i in range(img.shape[0]):
				for j in range(raio,img.shape[1]-raio):
					for k in range(raio,img.shape[2]-raio):
						rec = img[i,j-raio:j+raio+1,k-raio:k+raio+1]
						img2[j-raio,k-raio] = np.sum(rec*filt[i])
		else:
			for i in range(filt.shape[0]):
				for j in range(int(raio),img.shape[0]-raio):
					for k in range(raio,img.shape[1]-raio):
						rec = img[j-raio:j+raio+1,k-raio:k+raio+1]
						img2[j-raio,k-raio] = np.sum(rec*filt[i])				
	else:
		if(len(img.shape) > 2):
			for i in range(img.shape[0]):
				for j in range(raio,img.shape[1]-raio):
					for k in range(raio,img.shape[2]-raio):
						rec = img[i,j-raio:j+raio+1,k-raio:k+raio+1]
						img2[i,j-raio,k-raio] = np.sum(rec*filt)
		else:
			for j in range(raio,img.shape[0]-raio):
					for k in range(raio,img.shape[1]-raio):
						rec = img[j-raio:j+raio+1,k-raio:k+raio+1]
						img2[j-raio,k-raio] = np.sum(rec*filt)
	return img2														 

def convLayer(img,C,filters):
	x = []
	if(C > 1):

		fmap =[conv(img,filters[k]) for k in range(C)]
	   
		fmap = relu(fmap)

	else:
		if (len(filters.shape) > 2):
			fmap =[conv(img,filters[k]) for k in range(filters.shape[0])]
		else:
			fmap =conv(img,filters)
		#for i in range(0, C):
		fmap = relu(fmap)
	
		
	return fmap



def lenet(img):
	img = convLayer(img,1,filtros[0,0])                   
	img= maxpool(img, 2) #maxpool                    
				  
	img = convLayer(img,2,filtros[0,1])                    
	img = maxpool(img,2) #maxpool
					
	img = convLayer(img,1,filtros[0,2])                    
	#totalmente conectada                   
	img = np.asarray([np.sum(img * i) for i in filtros[0,3]])
	img = relu(img)
					
	img = np.asarray([np.sum(img * i) for i in filtros[0,4]])
				   
									  
	img = softmax(img)
	return img

img_train = mnist.train_images()
img_train_labels = mnist.train_labels()
img_train = batchsep(img_train,32)
img_train_labels = batchsep2(img_train_labels,32)
img = img_train
imgl = img_train_labels
learning_rate = 0.0001
h = 0.0001
img =padd(img_train,2)

pes1 = [he(5,5,6) for i in range(1)]
pes1 = np.asarray(pes1)
pes2 = []

pes2 = [[he(5,5,1) for _ in range(1)] for _ in range(2)]
pes2 = np.asarray(pes2)

pes3 = [[he(5,5,2) for _ in range(2)] for _ in range(1)]
pes3 = np.asarray(pes3)

pes5 = he(84,1,1)
pes6 = he(10,1,84)

dpes1 = np.zeros((pes1.shape[0],pes1.shape[1],pes1.shape[2]))
dpes2 = np.zeros((pes2.shape[0],pes2.shape[1],pes2.shape[2],pes2.shape[3]))
dpes3 = np.zeros((pes3.shape[0],pes3.shape[1],pes3.shape[2],pes3.shape[3]))
dpes5 = np.zeros(len(pes5))
dpes5.shape = (len(pes5),1)
dpes6 = np.zeros(len(pes6))
dpes6.shape = (len(pes6),1)
filtros = []

filtros.append([pes1,pes2,pes3,pes5,pes6])
dfiltros = []

dfiltros.append([dpes1,dpes2,dpes3,dpes5,dpes6])
filtros = np.asarray(filtros)
dfiltros = np.asarray(dfiltros)
l = 0
ct = 0
erros = [0]*32
erroslayer = [0]*32
erroslayernovo = [0]*32
saidas= [0]*32


def test():
	l =0
	for imgt in range(img.shape[0]):
		if (imgt == 32):
			for i in erros:
				print(i)
			
		for imgb in img[imgt]:
			
			imgb = lenet(imgb)
			igual = list(imgb).index(max(imgb))

			if( igual != imgl[imgt][l]):
				if(not(np.isfinite(-np.log(imgb[imgl[imgt][l]])))):
						  print('SIGH')
				erros[imgt] += 1
			erroslayer[l] = -np.log(imgb[imgl[imgt][l]])
		
			l = l+1    
		l = 0
		print(erros[imgt])
		for k in range(len(filtros[0])):
			if(k > 0):
				print(np.mean(erroslayernovo))
			for s in range(len(filtros[0,k].flat)):
				filtros[0,k].flat[s] +=h
			   
				for imgb in img[imgt]:

					imgb = lenet(imgb)            
					igual = list(imgb).index(max(imgb))

					if( igual != imgl[imgt][l]):
						if(not(np.isfinite(-np.log(imgb[imgl[imgt][l]])))):
							print('WHY?')
					erroslayernovo[l] = -np.log(imgb[imgl[imgt][l]])
	   				
					l = l+1
				filtros[0,k].flat[s] -=h	    
				l = 0
			   
				dfiltros[0,k].flat[s] = (np.mean(erroslayernovo) -np.mean(erroslayer) )/h    
			for k in range(5):
				filtros[0,k] -=learning_rate*dfiltros[0,k]
'''	
	x = [[5,1],[-5,10]]
	y = [[6,3],[6,-4]]
	z = [[7,-5],[7,-6]]
	k = [[-8,-7],[8,9]]
	s = []
	s.append(x)
	s.append(y)
	s.append(z)
	s.append(k)
	#print(s)
	s = np.asarray(s)
	d = [3.2,5.1,-1.7]
	print(softmax(d))
	#print(s.shape)
	l = batchsep(s,2)
	#print(l)
	#print(relu(s))	
'''

'''
def test():
	x = np.array([[  36.,  37.], [ 366., 174.], [ 115.,  59.],
				  [ 162.,  87.], [  38.,  90.], [ 258., 173.],
				  [ 224., 108.], [  78., 170.], [  72.,  43.]])
	print(len(x))
'''
test()			
