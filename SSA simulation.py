import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time

n = 256    #dimesions of screen or SLM
m = 256    #dimesions of screen or SLM

#Transmission Matrix
def randU(l,p):
    Are = np.random.randn(l,p)
    Aim = np.random.randn(l,p)
    A = Are + 1j*Aim
    Q,R = np.linalg.qr(A)
    B = np.dot(Q,np.diag(np.diag(R)/np.diag(abs(R))))
    return B

def Ei(l,p):
    A = np.random.randn(l,p)
    Q,R = np.linalg.qr(A)
    D = np.dot(Q,np.diag(np.diag(R)/np.diag(abs(R))))
    return D 
    



T = randU(n,m)
ei1 = Ei(n,m)
def intensity_matrix(A):
    b = np.zeros((n,m,3))
    a = A
    a = np.floor(np.abs(a)*255)
    b[:,:,0] = a
    b = b.astype(np.int32)
    return b
ef = np.matmul(T,ei1)
q1 = intensity_matrix(ef)

target = n//2
def target_intensity(Y):
    I = np.floor(sum(np.abs(Y[target:target+1,m//2:m//2+1]))**2*255)
    return I[0]

print('i0',target_intensity(ef))


phase = np.ones((n,m))
i0 = target_intensity(ef)
ei = ei1
t1 = time.time()
for i in range(n):
    for k in range(m):
        ei[i][k] = -1.0*ei[i][k]
        ef = np.matmul(T,ei1)
        i1 = target_intensity(ef)
        ei[i][k] = -1.0*ei[i][k]

        if i1 > i0:
            phase[i][k] = -1

        else:
            pass
t2 = time.time()
print('Time Taken',t2-t1)

for f in range(n):
    for g in range(m):
        ei[f][g] = phase[f][g]*ei[f][g]

ef = np.matmul(T,ei1)
q2 = intensity_matrix(ef)
print('i1',target_intensity(ef))


rows = 1
columns = 2
fig = plt.figure(figsize=(10, 7))
fig.add_subplot(rows, columns, 1)            
plt.imshow(q1)
plt.title("First")
fig.add_subplot(rows, columns, 2)
plt.imshow(q2)
plt.title("Second")
plt.show()
plt.close()