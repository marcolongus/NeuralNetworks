import numpy as np
import torch

device = torch.device("cpu")
dtype = torch.float

p = 64 #patrones

N_in  = 1000 #neuronas en la primera capa
N_h   = 100 #neuronas en la capa oculta
N_out = 10 #neuronas en la capa de salida

#Conjunto de entrenamiento aleatorio

x = torch.randn(p,N_in, device = device, dtype = dtype)
y = torch.randn(p,N_out, device = device, dtype = dtype)

#Matrices de pesos aleatorias.

w1 = torch.randn(N_h,N_in, device = device, dtype = dtype)
w2 = torch.randn(N_out,N_h, device = device, dtype = dtype)

learning_rate =1e-6

for i in range(501):

	#Foward pass:

	h1 = x.mm(w1.T)
	v  = h1.clamp(min=0) 
	y_pred  = v.mm(w2.T)

	#compute and print loss

	y_dif = y - y_pred
	loss = 0.5*y_dif.pow(2).sum().item()

	if(i%100==0):
		print(i,loss)

		
	#Backprop
	#Gradiente de w2

	grad_w2 = learning_rate*(y_dif.T.mm(v))

	#Gradiente de w1

	mat  = y_dif.mm(w2)
	grad = h1.clone()
	grad[h1<0] = 0
	grad[h1>0] = 1  

	mult = mat*grad

	grad_w1 = learning_rate*(mult.T.mm(x))

	#Acutalización

	w2 += grad_w2
	w1 += grad_w1

