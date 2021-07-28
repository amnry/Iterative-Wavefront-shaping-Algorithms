import numpy as np

import matplotlib.pyplot as plt
import time


n = 32    #dimesions of screen or SLM
m = 32    #dimesions of screen or SLM
x_rate = 0.5
N = 8
lamb = 1250

#Transmission Matrix
def randU(p):
    Are = np.random.randn(p,p)
    Aim = np.random.randn(p,p)
    A = Are + 1j*Aim
    Q,R = np.linalg.qr(A)
    B = np.dot(Q,np.diag(np.diag(R)/np.diag(abs(R))))
    return B

def Ei(l,p):
    A = np.random.randn(l,p)
    Q,R = np.linalg.qr(A)
    D = np.dot(Q,np.diag(np.diag(R)/np.diag(abs(R))))
    return D 
    
    
'''def multiply(A,B):
    ca = len(A[0])
    ra = len(A)
    rb = len(B)
    cb = len(B[0])
    C = np.zeros((ra,cb),dtype = np.complex128)
    for i in range(ra):
        for s in range(cb):
            for k in range(ca):
                C[i,s] = C[i,s] + np.dot(A[i,k],B[k,s])
    return C'''
def sortFirst(val):
    return val[0]

'''def sorting(A,B):
    C = zip(A,B)
    C =list(C)
    C.sort(key=sortFirst,reverse = True)
    C_sorted = [x for y,x in C]
    return C_sorted'''

T = randU(n)
ei1 = Ei(n,m)
'''def intense_matrix(A,B):
    b = np.zeros((n,m,3))
    Ef = np.matmul(B,A)
    a = np.matmul(Ef,Ef)
    a = np.floor(np.abs(a)*255)
    b[:,:,0] = a..............
    b = b.astype(np.int32)
    return b'''

def intensity_matrix(A):
    b = np.zeros((n,m,3))
    a = np.square(A)
    a = np.floor(np.abs(a)*255)
    b[:,:,0] = a
    b = b.astype(np.int32)
    return b
ef = np.matmul(T,ei1)
#q1 = intensity_matrix(ef)
#q1,ef =  intensity_matrix(ei1,T)
target = n//2
def target_intensity(Y):
    I  = np.floor((np.abs(Y[target,m//2])**2)*255)
    return I
'''def target_intense(Y):
    I = math.log(np.sum(Y[target:target+1,m//2:m//2+1,:]))
    return I'''

print('i1',target_intensity(ef))
pop0 = np.random.randint(0, 2, (N,n,m))
pop1 = np.ones((N,n,m),dtype=int)
pop = pop0*2-pop1
max_fitness =[target_intensity(ef)]
steps = [0]
t1 = time.time()
for s in range(1,200):
    steps.append(s)
    fitness = []
    for i in range(len(pop)):
        ei1 = ei1*pop[i]
        ef = np.matmul(T,ei1)
        fitness.append(target_intensity(ef))
        ei1 = ei1*pop[i]
    max_fitness.append(max(fitness))
    #print('a',len(pop),fitness)
    #sort
    pop = [x for _, x in sorted(list(zip(fitness,pop)),key=sortFirst,reverse = True)]
    '''pop = sorting(fitness,pop)'''
    pop = pop[:int(x_rate*N)]
    '''print('a',len(pop),fitness)'''
    '''ei1 = ei1*pop[0]
    ef = np.matmul(ei1,T)'''
    #Mating
    if s != 199:
        crossover = n//2
        e =  crossover
        def mate(ma,pa):
            offspring_1 = np.concatenate((ma[:e],pa[(n - e):]))
            offspring_2 = np.concatenate((pa[:e],ma[(n - e):]))
            return [offspring_1, offspring_2]
        for j in range(0,N,2):
            pop = np.concatenate((pop,mate(pop[j],pop[j+1])))

    #Mutations
    
        #mutation_rate = (-0.072 + 0.1)*np.exp(s/lamb) + 0.1
        if max(fitness)<= max_fitness[-1]:
            mutation_rate = 0.2
            #print(mutation_rate)
        else:    
            mutation_rate = 0.1    
        mu = mutation_rate
        mr = []
        mc = []
        mh = []
        for k in range(int(mu*N*n*m)):
            mc.append(np.random.randint(0,m))
            mr.append(np.random.randint(0,n))
            mh.append(np.random.randint(0,N))
        for z in range(len(mr)):
            pop[mh[z]][mr[z]][mc[z]] = pop[mh[z]][mr[z]][mc[z]] - 2*np.sign(pop[mh[z]][mr[z]][mc[z]])
    else:
        break
        
    print(s)
t2 = time.time()
ei1 = ei1*pop[0]
ef = np.matmul(T,ei1)
#q2 = intensity_matrix(ef)
print('Time Taken',t2-t1)
print('i2',target_intensity(ef))
'''print(ei1)'''

rows = 1
columns = 2
fig, ax = plt.subplots(1,2,figsize=(15,15))
circle2 = plt.Circle((n//2, m//2), 0.7,fill=False,edgecolor='green')
ax[0].add_patch(circle2)
#ax[0].imshow(q1)
ax[0].title.set_text('First')
circle3 = plt.Circle((n//2, m//2), 0.7,fill=False,edgecolor='green')
ax[1].add_patch(circle3)
#ax[1].imshow(q2)
ax[1].title.set_text('Second')
plt.show()
plt.close()

plt.plot(steps,max_fitness)
plt.xlabel('Steps')
plt.ylabel('Intensity at target')
plt.show()
plt.close()
'''for i in range(n):
    for k in range(m):
        if pop[0][i][k] == -1:
            pop[0][i][k]=0
        else:
            continue
pop[0] = pop[0]*255
l = np.zeros((n,m,3))
l[:,:,0] = pop[0]
plt.imshow(l)
plt.show()'''
