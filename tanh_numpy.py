import numpy as np
from activation import g

#Parametros de la red

p = 5

N_in  = 8
N_h   = 16
N_out = 8

#Generamos el conjunto de entrenamiento

x = np.random.randn(p,N_in)
y = np.random.randn(p,N_out)

#Generamos las matrices de pesos

w1 = np.random.randn(N_h,N_in)
w2 = np.random.randn(N_out,N_h)

learning_rate = 1e-6

for i in range(500):

	#foward pass:

	h1 = x.dot(w1.T)
	v  = g(h1)
	h2 =v.dot(w2.T)
	y_pred = g(h2)

	y_dif = y_pred-y

	#compute and print loss

	loss = 0.5*np.square(y_dif).sum()

	if (i%100 == 0):
		print(i, loss)


	#Backpropagation

	#w2


	T = np.ones(shape=(p,N_out)) - np.power(y_pred,2)
	M_yt = np.multiply(y_dif,T)

	w2_grad = learning_rate*M_yt.T.dot(v)

	
	#w1

	T2 = np.ones(shape=(p,N_h)) - np.power(v,2)
	m  = M_yt.dot(w2)
	M_mt2 = np.multiply(m,T2)

	w1_grad = learning_rate*M_mt2.T.dot(x)

	#Acutalización de las matrices de pesos:

	w1 += w1_grad
	w2 += w2_grad

	

