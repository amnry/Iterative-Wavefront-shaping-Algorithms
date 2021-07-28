import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pypylon import pylon
import time
import cv2

n = 1280    #dimesions of screen or SLM
m = 1024    #dimesions of screen or SLM

pixels= 64
nn = n//pixels     # dimension of phase mask
mm= m//pixels      # dimension of phase mask

xx = 50      # x coordinate shift of target spot from centre
yy = 50      # y coordinate shift of target spot from centre

target_size = 5     # side of a target spot which is taken in form of a square


def cam(number):
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    camera.ExposureTime = 220
    # Set Exposure time- 59 to 50000
    camera.BalanceRatioSelector = 'Red'               # Balance Ratio Selector- Red/Blue/Green
    camera.BalanceRatio = 1.0                       # Balance Ratio varrying from- 0 to 15.98
    camera.BalanceRatioSelector = 'Green'               # Balance Ratio Selector- Red/Blue/Green
    camera.BalanceRatio = 0                       # Balance Ratio varrying from- 0 to 15.98
    camera.BalanceRatioSelector = 'Blue'               # Balance Ratio Selector- Red/Blue/Green
    camera.BalanceRatio = 0                           # Balance Ratio varrying from- 0 to 15.98
    camera.ExposureAuto = 'Off'                       # ExposureAuto- Off /Once /Continous
    camera.LightSourcePreset = 'Off'        # Light Source Preset- Daylight5000K /Off /Tungsten2800K /Daylight6500K
    #Triggering thecamera
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
            images.append(img)

    grab.Release()
    camera.StopGrabbing()
    camera.Close()
    return images

def slm(MASK):
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
    cv2.waitKey (500)
    IMG = cam(1)
    return IMG[0]


def disp(MASK):
    b = np.zeros((nn,mm,3))
    b[:,:,0] = MASK
    b[:,:,1] = MASK
    b[:,:,2] = MASK
    b = b.astype(np.int32)
    uint_img = np.array(b).astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_BGR2GRAY)
    # Show full screen
    px = 1920
    cv2.namedWindow ('screen', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty ('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow('screen', px,0)
    cv2.imshow ('screen', grayImage)
    cv2.waitKey (200)
    return


def target_intensity(Y):
    I  = np.sum(Y[n//2 + xx:n//2 + xx + target_size,m//2 - yy:m//2 - yy + target_size])
    return I/(target_size)**2

def invert(value):
    if value == 0:
        p = 255
    else:
        p = 0
    return p    
    
pop0 = np.random.randint(0, 2,(1,nn,mm))
pop = pop0*255    # random phase mask of 0 and 255 of dimensions nn x mm

img = slm(pop)
T1 = time.time()
img = slm(pop)
print('t',target_intensity(img))
t0 = target_intensity(img)


fitness = [t0]
fitness1 = []
i = 0
while i < nn:
    k = 0
    while k< mm:
        pop[0][i][k] = invert(pop[0][i][k])
        img1 = slm(pop)
        t1 = target_intensity(img1)
        cv2.destroyAllWindows()

    
            
        if t1>t0 and t1-t0 < 0.2*t0:
            t0 = t1
            fitness.append(t1)
            fitness1.append(t1)
            k = k + 1
        elif t1-t0>0.2*t0:
            k = k
        else:
            pop[0][i][k] = invert(pop[0][i][k])
            fitness.append(t0)
            fitness1.append(t1)
           
            k = k + 1
            
    disp(pop)
    i = i + 1
    
        
    
img1 = slm(pop)    
print('t1',target_intensity(img1))
T2 = time.time()
l = np.arange(0,nn*mm + 1,1)
plt.plot(l,fitness)
plt.plot(l[1:],fitness1)
plt.show()
print('Time taken: ',T2-T1)
