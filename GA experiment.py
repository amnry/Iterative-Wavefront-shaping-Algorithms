import numpy as np
from pypylon import pylon
import matplotlib.pyplot as plt
import time
import cv2

n = 1280    #dimesions of screen or SLM
m = 1024    #dimesions of screen or SLM

x_rate = 0.5   # Fraction of population selected for next generation
N = 8          # Initial Population size
pixels = 32    
nn = n//pixels  # dimension of phase mask
mm = m//pixels   # dimension of phase mask

gen = 30        # number of generations

xx = 100    # x coordinate shift of target spot from centre
yy = 100    # y coordinate shift of target spot from centre
target_size = 5     # side of a target spot which is taken in form of a square

def cam(number):
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    camera.ExposureTime = 220                           # Set Exposure time- 59 to 50000
    camera.BalanceRatioSelector = 'Red'               # Balance Ratio Selector- Red/Blue/Green
    camera.BalanceRatio = 1.0                       # Balance Ratio varrying from- 0 to 15.98
    camera.BalanceRatioSelector = 'Green'               # Balance Ratio Selector- Red/Blue/Green
    camera.BalanceRatio = 0                       # Balance Ratio varrying from- 0 to 15.98
    camera.BalanceRatioSelector = 'Blue'               # Balance Ratio Selector- Red/Blue/Green
    camera.BalanceRatio = 0                           # Balance Ratio varrying from- 0 to 15.98
    camera.ExposureAuto = 'Off'                       # ExposureAuto- Off /Once /Continous
    camera.LightSourcePreset = 'Off'        # Light Source Preset- Daylight5000K /Off /Tungsten2800K /Daylight6500K
    
    camera.TriggerActivation = 'RisingEdge'           # Trigger Activation- RisingEdge /FallingEdge
    camera.TriggerDelay = 0                           # Trigger Delay
    camera.TriggerMode = 'On'                         # Set Trigger mode- On/Off
    camera.TriggerSelector = 'FrameStart'             # Trigger Selector- FrameStart /FrameBurstStart
    camera.TriggerSource = 'Line3'                    # Trigger Source- Line1 /Software /Line3 /Line4 /SoftwareSignal1 /SoftwareSignal2 /SoftwareSignal3
    
    camera.AcquisitionStatusSelector = 'FrameTriggerWait'

    images = []
    for i in range(number):
        camera.RegisterConfiguration(pylon.ConfigurationEventHandler(), pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_Delete) 
        grab = camera.RetrieveResult(200, pylon.TimeoutHandling_Return)
        if grab.GrabSucceeded():

            img = grab.GetArray()
            images = img

   
    grab.Release()
    camera.StopGrabbing()
    camera.Close()
    return images


# Display image on SLM
def slm(MASK,pause):
    b = np.zeros((nn,mm,3))
    b[:,:,0] = MASK
    b[:,:,1] = MASK
    b[:,:,2] = MASK
    b = b.astype(np.int32)
    uint_img = np.array(b).astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_BGR2GRAY)
    # Show full screen
    px = 1920 + 1920
    cv2.namedWindow ('screen', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty ('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow('screen', px,0)
    cv2.imshow ('screen', grayImage)
    cv2.waitKey (pause)
    IMG = cam(1)
    return IMG

    

def sortFirst(val):
    return val[0]


def target_intensity(Y):
    I  = np.sum(Y[n//2 + xx:n//2 + xx + target_size,m//2 - yy:m//2 - yy + target_size])
    return I/(target_size)**2

pop0 = np.random.randint(0, 1,(1,nn,mm))
pop = pop0*255      # black screen display on slm
img1 = slm(pop,1000)


print('i1',target_intensity(img1))
max_fitness =[target_intensity(img1)]
steps = [0]
pop0 = np.random.randint(0, 2,(N,nn,mm))     
pop = pop0*255       # random phase mask of 0 and 255 of dimensions nn x mm
t1 = time.time()
for s in range(1,gen):
    steps.append(s)
    fitness = []
    for i in range(len(pop)):
        image = slm(pop[i],500)
        fitness.append(target_intensity(image))
    max_fitness.append(max(fitness))
   
    pop = [x for _, x in sorted(list(zip(fitness,pop)),key=sortFirst,reverse = True)]    # ranking of phase masks
   
    pop = pop[:int(x_rate*N)]     # top half of the population selected for next generation
    
    #Mating
    if s != gen-1:
        crossover = nn//2
        e =  crossover
        def mate(ma,pa):
            offspring_1 = np.concatenate((ma[:e],pa[(nn - e):]))
            offspring_2 = np.concatenate((pa[:e],ma[(nn - e):]))
            return [offspring_1, offspring_2]
        for j in range(0,N,2):
            pop = np.concatenate((pop,mate(pop[j],pop[j+1])))

    #Mutations
    
        if max(fitness)<= max_fitness[-1]:
            mutation_rate = 0.07
        else:    
            mutation_rate = 0.001    
        mu = mutation_rate
        mr = []
        mc = []
        mh = []
        for k in range(int(mu*N*nn*mm)):
            mc.append(np.random.randint(0,mm))
            mr.append(np.random.randint(0,nn))
            mh.append(np.random.randint(0,N))
        for z in range(len(mr)):
            if pop[mh[z]][mr[z]][mc[z]] == 0:
                pop[mh[z]][mr[z]][mc[z]] = 255
            else:
                pop[mh[z]][mr[z]][mc[z]] = 0
    else:
        break
        
    print(s)
t2 = time.time()
image = slm(pop[0],500)

print('Time Taken',t2-t1)
print('i2',target_intensity(image))
plt.plot(steps,max_fitness)
plt.show()
plt.close()

rows = 1
columns = 2
fig, ax = plt.subplots(1,2,figsize=(15,15))
circle2 = plt.Circle((n//2 + xx, m//2 - yy), 15,fill=False,edgecolor='green')
ax[0].add_patch(circle2)
ax[0].imshow(img1)
ax[0].title.set_text('First')
circle3 = plt.Circle((n//2 + xx, m//2 - yy), 15,fill=False,edgecolor='green')
ax[1].add_patch(circle3)
ax[1].imshow(image)
ax[1].title.set_text('Second')
plt.show()
plt.close()
