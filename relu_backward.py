#con backward
import torch

p = 64 #patrones

N_in  = 1000 #neuronas en la primera capa
N_h   = 100 #neuronas en la capa oculta
N_out = 10 #neuronas en la capa de salida

#Conjunto de entrenamiento aleatorio

x = torch.randn(p,N_in)
y = torch.randn(p,N_out)

#Matrices de pesos aleatorias.

w1 = torch.randn(N_h,N_in, requires_grad=True)
w2 = torch.randn(N_out,N_h, requires_grad=True)

learning_rate=1e-6

for i in range(501):

	#foward pass:

	y_pred = x.mm(w1.T).clamp(min=0).mm(w2.T)

	loss = 0.5*(y_pred -y).pow(2).sum()

	if (i%100==0):
		print(i, loss.item())


	loss.backward()

	with torch.no_grad():
		w1 -= learning_rate*w1.grad
		w2 -= learning_rate*w2.grad

		w1.grad.zero_()
		w2.grad.zero_()
