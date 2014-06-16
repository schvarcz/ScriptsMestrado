from naoqi import ALProxy
import math
import pygame
from pygame.locals import *
import time
import os

totalx = 0
totaly = 0
totalz = 0

PORT = 9559
robotIp = "192.168.0.1" #real
#robotIp = "127.0.0.1"  #simulado

#Proxy creation
motion = ALProxy("ALMotion", robotIp, PORT)
posture = ALProxy("ALRobotPosture", robotIp, PORT)


# Connect to ALSonar module.
sonarProxy = ALProxy("ALSonar", robotIp, 9559)

# Subscribe to sonars, this will launch sonars (at hardware level) and start data acquisition.
sonarProxy.subscribe("myApplication")

#Now you can retrieve sonar data from ALMemory.
memoryProxy = ALProxy("ALMemory", robotIp, 9559)

start = None
elapsed = None

sx = 0
vx = 0
t0 = 0

tt = 0
contagem = 0



fo = open("sonares.txt", "w")
fo2 = open("accelerometer.txt", "w")
fo3 = open("gyrometer.txt", "w")


               
def main(robotIp):
        
        total = 0
        pygame.init()
        screen = pygame.display.set_mode((150, 50))
        pygame.display.set_caption('Basic Pygame program')
        
        motion.setStiffnesses("Body", 1.0)
        
        
        ## gravacao
        
#        videoRecorderProxy = ALProxy("ALVideoRecorder", robotIp, PORT)

        # This records a 320*240 MJPG video at 10 fps.
        # Note MJPG can't be recorded with a framerate lower than 3 fps.

#        videoRecorderProxy.startRecording("./", "video01", True)

        n = 1
        move = 0
        
        if posture.getPostureFamily() != "Standing":
            posture.goToPosture("Stand", 1.0)
        
        while posture.getPostureFamily() != "Standing":
            print posture.getPostureFamily()
        
        motion.moveInit()
        
        action = None
        oldAction = None
        
        t0 = time.clock()

        
        if posture.getPostureFamily() == "Standing":
            while n:
                
                for event in pygame.event.get():
                    if event.type == QUIT:
                        return
                    else:
                        keys = pygame.key.get_pressed()  #checking pressed keys
                        
                        if keys[K_w]:
                            print "AndarFrente"
                            action = AndarFrente
                            
                        elif keys[K_s]:
                            print "AndarTras"
                            action = AndarTras
                                 
                        elif keys[K_d]:
                            print "GirarDireita"
                            action = GirarDireita
                                   
                        elif keys[K_a]:
                            print "GirarEsquerda"
                            action = GirarEsquerda
                                
                        elif keys[K_y]:
                            print "Levantar"
                            action = Levantar
          
                        elif keys[K_h]: 
                            print "Sentar"
                            action = Sentar
                                      
                        elif keys[K_ESCAPE]:
                            print "SairPrograma"
                            action = SairPrograma
                            
                        elif keys[K_SPACE]:
                            print "Parar"
                            action = Parar
                
                #PrintGyrometer()
                #GetSonar()
                #GetAccelerometer()
                #GetGyrometer()
                
                if action != None:                  
                    if action != oldAction:
                        motion.moveInit()
                        
                        temp = t0
                        t0 = time.clock()
                        elapsed = t0 - temp
                        #print elapsed
                        #fo2.write("*****************************")
                        #fo2.write("tempo: " + str(elapsed) + '\n')
                        action()
                        oldAction = action
                   
                   

def Parar():
    motion.moveToward(0, 0, 0)

                   
def AndarFrente():  
    motion.moveToward(0.5, 0, 0)

                    
def AndarTras():       
    motion.moveToward(-0.5, 0, 0)
 

def GirarEsquerda():
    motion.moveToward(0, 0, math.pi/10)


def GirarDireita():
    motion.moveToward(0, 0, math.pi/-10)

def SairPrograma():   
    motion.killMove()
#    videoRecorderProxy.stopVideoRecord()
    posture.goToPosture("Sit", 1.0)
    motion.setStiffnesses("Body", 0)
    fo.close()
    fo2.close()
    fo3.close()
    print "Distancia:" + str(sx)
    tirar = tt / contagem
    print tirar
    n = 0;

def Sentar():
    posture.goToPosture("Sit", 1.0)

def Levantar():
    posture.goToPosture("Stand", 1.0)

 
def GetSonar():
    # Get sonar left first echo (distance in meters to the first obstacle).
    fo.write('Left:'+' ')  
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value1"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value2"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value3"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value4"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value5"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value6"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value7"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value8"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value9"))+'\n') 

    # Same thing for right.
    fo.write('Right:'+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value1"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value2"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value3"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value4"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value5"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value6"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value7"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value8"))+' ')
    fo.write(str(memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value9"))+'\n')

    # Please read Sonar ALMemory keys section if you want to know the other values you can get.

def GetAccelerometer():

    global vx
    global sx
    global t0
    
    global tt
    global contagem

    fo2.write('Accelerometer:'+'     ')
    fo2.write(str(memoryProxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerX/Sensor/Value"))+'      ')
    #fo2.write(str(memoryProxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerY/Sensor/Value"))+'      ')
    #fo2.write(str(memoryProxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerZ/Sensor/Value"))+'      ')
    

    alfa = memoryProxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerX/Sensor/Value") - 0.65
    
    tt = tt + alfa
    contagem = contagem+1
    
    if alfa > 0.25 or alfa <-0.90:
        alfa = alfa
    
    #elif:
        alfa = 0
        
    temp = t0
    t0 = time.clock()
    elapsed = t0 - temp
    
    fo2.write("tempo: " + str(elapsed) + '\n')
    
    
    sx = sx + vx * elapsed + (alfa * (elapsed ** 2)) / 2
    vx = vx + alfa * elapsed
     


    
def GetGyrometer():
    fo3.write('Gyrometer:'+' ')
    #fo3.write(str(memoryProxy.getData("Device/SubDeviceList/InertialSensor/GyrRef/Sensor/Value"))+' ')
    fo3.write(str(memoryProxy.getData("Device/SubDeviceList/InertialSensor/GyrX/Sensor/Value"))+' ')
    fo3.write(str(memoryProxy.getData("Device/SubDeviceList/InertialSensor/GyrY/Sensor/Value"))+'\n')

def PrintGyrometer():
        #os.system('cls')
        
        global totalx
        global totaly
        global totalz
        
        gyrx = memoryProxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeX/Sensor/Value") 
        gyry = memoryProxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeY/Sensor/Value") 
        gyrz = memoryProxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeZ/Sensor/Value") 
        
           
        
        #print memoryProxy.getData("Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value")
        #print memoryProxy.getData("Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value")         

        
       # print type(gyrz)
        
       # print str(gyrx) + " rad/s"
       # print str(gyry) + " rad/s"
       # print str(gyrz) + " rad/s"
        
        if gyrx > 0.012:
            totalx = totalx + float(gyrx)
        if gyrz > 0.012:
            totalz = totalz + float(gyrz)
        if gyry > 0.012:
            totaly = totaly + float(gyry)
        
       # print "Total X: " + str(totalx)
       # print "Total Y: " + str(totaly)
       # print "Total Z: " + str(totalz)


    
if __name__ == "__main__":
    
    main(robotIp)
