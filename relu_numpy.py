import numpy as np

p = 64 #patrones

N_in  = 1000 #neuronas en la primera capa
N_h   = 100 #neuronas en la capa oculta
N_out = 10 #neuronas en la capa de salida

#Conjunto de entrenamiento aleatorio

x = np.random.randn(p,N_in)
y = np.random.randn(p,N_out)

#Matrices de pesos aleatorias.

w1 = np.random.randn(N_h,N_in)
w2 = np.random.randn(N_out,N_h)

learning_rate =1e-6

for i in range(100000):

	#Foward pass:

	h1 = x.dot(w1.T)
	v  = np.maximum(h1,0) 
	y_pred  = v.dot(w2.T)

	#compute and print loss

	y_dif = y - y_pred
	loss = 0.5*np.square(y_dif).sum()

	if (i%1000 == 0):
		print(i,loss)
		
	#Backprop
	#Gradiente de w2

	grad_w2 = learning_rate*(y_dif.T.dot(v))

	#Gradiente de w1

	mat  = y_dif.dot(w2)
	grad = h1.copy()
	grad[h1<0] = 0
	grad[h1>0] = 1  

	mult = np.multiply(mat,grad)

	grad_w1 = learning_rate*(mult.T.dot(x))

	#Acutalización

	w2 += grad_w2
	w1 += grad_w1

