#Part 2 production of data and accuracy test

#416
#507
#548

import sys
sys.path.insert(0, '../../Utilities/')
import scipy.io
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt

Lx=1.0
Nx=200
eps=1/64
#Nx=30
#eps=1.0/8.0
Nv=16

dx = Lx/Nx
x=np.zeros([Nx,1],dtype=float)
for I in range(0,Nx):
    x[I,0] =-Lx/2 + dx*I


TTdata = scipy.io.loadmat('Periodic_data.mat')
v = TTdata['v']
wv=TTdata['wv']





Nv=16

#v=np.zeros([Nv,1])
#wv=np.zeros([Nv,1])

"""
v[0,0]=-0.99518721999702136018
v[1,0]=-0.9747285559713094981984
v[2,0]=-0.9382745520027327585237
v[3,0]=-0.8864155270044010342132
v[4,0]=-0.820001985973902921954
v[5,0]=-0.7401241915785543642438
v[6,0]=-0.648093651936975569253
v[7,0]=-0.545421471388839535658
v[8,0]=-0.4337935076260451384871
v[9,0]=-0.3150426796961633743868
v[10,0]=-0.191118867473616309159
v[11,0]=-0.064056892862605626085
v[12,0]=0.064056892862605626085
v[13,0]=0.1911188674736163091586
v[14,0]=0.3150426796961633743868
v[15,0]=0.4337935076260451384871
v[16,0]=0.5454214713888395356584
v[17,0]=0.6480936519369755692525
v[18,0]=0.7401241915785543642438
v[19,0]=0.820001985973902921954
v[20,0]=0.8864155270044010342132
v[21,0]=0.9382745520027327585237
v[22,0]=0.9747285559713094981984
v[23,0]=0.99518721999702136018

wv[0,0]=0.0123412297999871995468
wv[1,0]=0.0285313886289336631813
wv[2,0]=0.0442774388174198061686
wv[3,0]=0.05929858491543678074637
wv[4,0]=0.07334648141108030573403
wv[5,0]=0.0861901615319532759172
wv[6,0]=0.0976186521041138882699
wv[7,0]=0.1074442701159656347826
wv[8,0]=0.1155056680537256013533
wv[9,0]=0.1216704729278033912045
wv[10,0]=0.1258374563468282961214
wv[11,0]=0.1279381953467521569741
wv[12,0]=0.1279381953467521569741
wv[13,0]=0.1258374563468282961214
wv[14,0]=0.121670472927803391204
wv[15,0]=0.115505668053725601353
wv[16,0]=0.1074442701159656347826
wv[17,0]=0.09761865210411388827
wv[18,0]=0.0861901615319532759172
wv[19,0]=0.073346481411080305734
wv[20,0]=0.0592985849154367807464
wv[21,0]=0.044277438817419806169
wv[22,0]=0.0285313886289336631813
wv[23,0]=0.0123412297999871995468
"""

"""
v[0,0]=-0.997263861849481563545
v[1,0]=-0.9856115115452683354002
v[2,0]=-0.9647622555875064307738
v[3,0]=-0.9349060759377396891709
v[4,0]=-0.8963211557660521239653
v[5,0]=-0.8493676137325699701337
v[6,0]=-0.794483795967942406963
v[7,0]=-0.7321821187402896803874
v[8,0]=-0.6630442669302152009751
v[9,0]=-0.5877157572407623290408
v[10,0]=-0.5068999089322293900238
v[11,0]=-0.421351276130635345364
v[12,0]=-0.3318686022821276497799
v[13,0]=-0.2392873622521370745446
v[14,0]=-0.1444719615827964934852
v[15,0]=-0.0483076656877383162348
v[16,0]=0.048307665687738316235
v[17,0]=0.1444719615827964934852
v[18,0]=0.2392873622521370745446
v[19,0]=0.33186860228212764978
v[20,0]=0.4213512761306353453641
v[21,0]=0.5068999089322293900238
v[22,0]=0.5877157572407623290408
v[23,0]=0.6630442669302152009751
v[24,0]=0.7321821187402896803874
v[25,0]=0.7944837959679424069631
v[26,0]=0.8493676137325699701337
v[27,0]=0.8963211557660521239653
v[28,0]=0.9349060759377396891709
v[29,0]=0.9647622555875064307738
v[30,0]=0.9856115115452683354002
v[31,0]=0.997263861849481563545




wv[0,0]=0.0070186100094700966004
wv[1,0]=0.0162743947309056706052
wv[2,0]=0.0253920653092620594558
wv[3,0]=0.0342738629130214331027
wv[4,0]=0.0428358980222266806569
wv[5,0]=0.050998059262376176196
wv[6,0]=0.0586840934785355471453
wv[7,0]=0.065822222776361846838
wv[8,0]=0.072345794108848506225
wv[9,0]=0.0781938957870703064717
wv[10,0]=0.0833119242269467552222
wv[11,0]=0.087652093004403811143
wv[12,0]=0.091173878695763884713
wv[13,0]=0.09384439908080456563918
wv[14,0]=0.0956387200792748594191
wv[15,0]=0.0965400885147278005668
wv[16,0]=0.0965400885147278005668
wv[17,0]=0.0956387200792748594191
wv[18,0]=0.0938443990808045656392
wv[19,0]=0.091173878695763884713
wv[20,0]=0.0876520930044038111428
wv[21,0]=0.083311924226946755222
wv[22,0]=0.078193895787070306472
wv[23,0]=0.072345794108848506225
wv[24,0]=0.065822222776361846838
wv[25,0]=0.0586840934785355471453
wv[26,0]=0.0509980592623761761962
wv[27,0]=0.0428358980222266806569
wv[28,0]=0.0342738629130214331027
wv[29,0]=0.0253920653092620594558
wv[30,0]=0.0162743947309056706052
wv[31,0]=0.0070186100094700966004
"""













rho = np.zeros([Nx,1],dtype=float)
for I in range(0,Nx):
#    rho[I,0] =20*x[I,0]*(x[I,0])*(x[I,0]-1.0)*np.sin(2.0*np.pi*x[I,0])**2*np.exp(-15.0*(x[I,0]-0.5)**4.0)
#    rho[I,0] =20*x[I,0]*(x[I,0])*(x[I,0]-1.0)*np.sin(2.0*np.pi*x[I,0])**2*np.exp(-25.0*(x[I,0]-0.5)**2.0)
#    rho[I,0] =20*x[I,0]*(x[I,0])*(x[I,0]-1.0)*np.sin(2.0*np.pi*x[I,0])**2*np.exp(-300.0*(x[I,0])**2.0)
#    rho[I,0] = 3*np.exp(-50.0*(x[I,0]-Lx/2)**2.0)
    rho[I,0] = 1+np.sin(2.0*np.pi*x[I,0])*np.exp(-300.0*(x[I,0])**2.0)
#    rho[I,0] = x[I,0]*np.exp(-10000.0*(x[I,0]-0.5)**4.0)
    
    
#rho[0,I]=rho[0,0]    
g = np.zeros([Nv,Nx],dtype=float)
gk1 = np.zeros([Nv,Nx],dtype=float)
gk2 = np.zeros([Nv,Nx],dtype=float)
gk3 = np.zeros([Nv,Nx],dtype=float)
rh1 = np.zeros([Nx,1],dtype=float)
rh2 = np.zeros([Nx,1],dtype=float)
rh3 = np.zeros([Nx,1],dtype=float)

sigmaA = np.zeros([1,Nx],dtype=float)
sigmaAg = np.zeros([Nx],dtype=float)
#sigmaArho = np.zeros([1,Nx+1],dtype=float)

sigmaS = np.ones([Nx],dtype=float)
for I in range(0,Nx):
    sigmaS[I]=1.0#(6+40*(x[I,0]-0.5)*np.exp(-500.0*(x[I,0]-Lx/2)**4.0))
#    sigmaS[I]=np.sin(2*np.pi*x[I,0])+3.0

G = np.zeros([Nx,1],dtype=float)
ss = np.zeros([Nx,1],dtype=float)
ssk1 = np.zeros([Nx,1],dtype=float)
ssk2 = np.zeros([Nx,1],dtype=float)
ssk3 = np.zeros([Nx,1],dtype=float)

#fL = v
#fR = np.zeros([Nv,1],dtype=float)

#UFg = np.zeros([Nv,Nx],dtype=float)
UFgk1 = np.zeros([Nv,Nx],dtype=float)
UFgk2 = np.zeros([Nv,Nx],dtype=float)
UFgk3 = np.zeros([Nv,Nx],dtype=float)
UFg = np.zeros([Nv,Nx],dtype=float)

PUFgk1 = np.zeros([Nv,Nx],dtype=float)
PUFgk2 = np.zeros([Nv,Nx],dtype=float)
PUFgk3 = np.zeros([Nv,Nx],dtype=float)
PUFg = np.zeros([Nv,Nx],dtype=float)
#PUFg = np.zeros([1,Nx],dtype=float)
Frho1 = np.zeros([Nv,Nx],dtype=float)
Frho2 = np.zeros([Nv,Nx],dtype=float)
Frho3 = np.zeros([Nv,Nx],dtype=float)
Frho = np.zeros([Nv,Nx],dtype=float)



N_s=3
A1=np.zeros([N_s,N_s],dtype=float)
c1=np.zeros([N_s,1],dtype=float)
w1=np.zeros([N_s,1],dtype=float)

A2=np.zeros([N_s,N_s],dtype=float)
c2=np.zeros([N_s,1],dtype=float)
w2=np.zeros([N_s,1],dtype=float)

#IMEX RK ARS(2,2,2)
gamma=1.0-np.sqrt(2.0)/2.0
delta=1.0-1.0/(2.0*gamma)
A1[1,0]=gamma
A1[2,0]=delta
A1[2,1]=1.0-delta
w1[0,0]=delta
w1[1,0]=1.0-delta
w1[2,0]=0.0

A2[1,1]=gamma
A2[2,1]=1.0-gamma
A2[2,2]=gamma
w2[0,0]=0.0
w2[1,0]=1.0-gamma
w2[2,0]=gamma










CFg = np.zeros([Nv,Nx],dtype=float)
PCFg = np.zeros([1,Nx],dtype=float)

gL = np.zeros([Nv,1],dtype=float)
gR = np.zeros([Nv,1],dtype=float)

gLk1 = np.zeros([Nv,1],dtype=float)
gRk1 = np.zeros([Nv,1],dtype=float)
gLk2 = np.zeros([Nv,1],dtype=float)
gRk2 = np.zeros([Nv,1],dtype=float)
gLk3 = np.zeros([Nv,1],dtype=float)
gRk3 = np.zeros([Nv,1],dtype=float)



dtt = (3*np.min(sigmaS)*dx**2/2+eps*dx)/(2.0*2.0)
#dt=2*10**-5/80000
dt=1*10**-7/2

dt=dx*dx/2

#dt=dtt/2
#dt=dtt
suggest=dtt/dt

#Tf=.0005

#nt=(Tf/dt)
nt=28*(64)

Tf=nt*dt

print(nt)
print(nt)
print(nt)
print(nt)

#nt*dt

#nt=400

#n1=3*np.min(sigmaS)*dx**2/2/(2.0*2.0)
#n2=eps*dx/(2.0*2.0)
#dt=np.min(n1,n2)



#NNN=6.0
#if n1>n2:
#    dt=n1
#    nt=int(NNN**2*8)
#else:
#    dt=n2
#    nt=int(NNN*8)
#tmax=nt*dt


Data_rho = np.zeros([Nx,int(nt)],dtype=float)
Data_g = np.zeros([Nv,Nx,int(nt)],dtype=float)

#Data_g = np.ones([Nv,Nx,int(nt)],dtype=float)
Data_idp = np.zeros([Nv,Nx,int(nt)],dtype=float)
Data_Frho = np.zeros([Nv,Nx,int(nt)],dtype=float)
pvdxg_data = np.zeros([Nx,int(nt)],dtype=float)

s = np.zeros([Nv,1],dtype=float)
ss = np.zeros([Nx,1],dtype=float)


Data_testsave = np.zeros([Nv,Nx,int(nt)],dtype=float)
Data_testsave2 = np.zeros([Nv,Nx,int(nt)],dtype=float)

gk1test=np.zeros([Nv,Nx,int(nt)],dtype=float)
UFgk1test=np.zeros([Nv,Nx,int(nt)],dtype=float)

#print('finish at')
#print(nt)
for tt in range(0,int(nt)):
    
    print(tt/nt*100)
    
    Data_g[:,:,tt]=g[:,:]
    Data_rho[:,tt]=rho[:,0]
    
    
    
    gk1=g
    rh1=rho            
            
    for I in range(0,Nv):
        if v[I]>0:
            for J in range(2,Nx-1):
                UFgk1[I,J] =v[I,0]*(2.0*gk1[I,J+1]+3.0*gk1[I,J]-6.0*gk1[I,J-1]+gk1[I,J-2])/(6.0*dx)           
#            UFgk1[I,2:Nx-2] =v[I,0]*(2.0*gk1[I,1:Nx-1]+3.0*gk1[I,2:Nx-2]-6.0*gk1[I,1:Nx-3]+gk1[I,0:Nx-4])/(6.0*dx)
            UFgk1[I,Nx-1] =v[I,0]*(2.0*gk1[I,0]+3.0*gk1[I,Nx-1]-6.0*gk1[I,Nx-2]+gk1[I,Nx-3])/(6.0*dx)
            UFgk1[I,1] = v[I,0]*(2.0*gk1[I,2]+3.0*gk1[I,1]-6.0*gk1[I,0]+gk1[I,Nx-1])/(6.0*dx)
            UFgk1[I,0] =v[I,0]*(2.0*gk1[I,1]+3.0*gk1[I,0]-6.0*gk1[I,Nx-1]+gk1[I,Nx-2])/(6.0*dx)          
        else:
            for J in range(1,Nx-2):
                UFgk1[I,J] =v[I,0]*(-1.0*gk1[I,J+2]+6.0*gk1[I,J+1]-3.0*gk1[I,J]-2.0*gk1[I,J-1])/(6.0*dx)
            UFgk1[I,0] = v[I,0]*(-1.0*gk1[I,2]+6.0*gk1[I,1]-3.0*gk1[I,0]-2.0*gk1[I,Nx-1])/(6.0*dx)
            UFgk1[I,Nx-2] =v[I,0]*(-1.0*gk1[I,0]+6.0*gk1[I,Nx-1]-3.0*gk1[I,Nx-2]-2.0*gk1[I,Nx-3])/(6.0*dx)
            UFgk1[I,Nx-1] =v[I,0]*(-1.0*gk1[I,1]+6.0*gk1[I,0]-3.0*gk1[I,Nx-1]-2.0*gk1[I,Nx-2])/(6.0*dx)
            
            
            

    for i in range(0,Nx):
        for j in range(0,Nv):
            s[j,0]=wv[j,0]*UFgk1[j,i]########Could this be a problem
        ssk1[i,0]=sum(s)/2#pvdxg, unextended to all v
    for j in range(0,Nv):
        for i in range(0,Nx):
            PUFgk1[j,i] = ssk1[i,0] #pvdxg extended to all v  
    

    for I in range(0,Nv):
        if v[I]>0:
#            for J in range(0,Nx-2):
            for J in range(1,Nx-2):
                Frho1[I,J] =v[I,0]*(-1.0*rh1[J+2,0]+6.0*rh1[J+1,0]-3.0*rh1[J,0]-2.0*rh1[J-1,0])/(6.0*dx)
            Frho1[I,0] = v[I,0]*(-1.0*rh1[2,0]+6.0*rh1[1,0]-3.0*rh1[0,0]-2.0*rh1[Nx-1,0])/(6.0*dx)
            Frho1[I,Nx-2] =v[I,0]*(-1.0*rh1[0,0]+6.0*rh1[Nx-1,0]-3.0*rh1[Nx-2,0]-2.0*rh1[Nx-3,0])/(6.0*dx)
            Frho1[I,Nx-1] =v[I,0]*(-1.0*rh1[1,0]+6.0*rh1[0,0]-3.0*rh1[Nx-1,0]-2.0*rh1[Nx-2,0])/(6.0*dx)
#                Frho1[I,J] = v[I,0]*(-1.0*rh1[0,J+2]+4.0*rh1[0,J+1]-3.0*rh1[0,J])/(2.0*dx)
#            Frho1[I,Nx-2] = v[I,0]*(-1.0*rh1[0,0]+4.0*rh1[0,Nx-1]-3.0*rh1[0,Nx-2])/(2.0*dx)
#            Frho1[I,Nx-1] =v[I,0]*(-1.0*rh1[0,1]+4.0*rh1[0,0]-3.0*rh1[0,Nx-1])/(2.0*dx)
        else:
            for J in range(2,Nx-1):
#            for J in range(2,Nx):
                Frho1[I,J] =v[I,0]*(2.0*rh1[J+1,0]+3.0*rh1[J,0]-6.0*rh1[J-1,0]+rh1[J-2,0])/(6.0*dx)
            Frho1[I,Nx-1] =v[I,0]*(2.0*rh1[0,0]+3.0*rh1[Nx-1,0]-6.0*rh1[Nx-2,0]+rh1[Nx-3,0])/(6.0*dx)
            Frho1[I,1] = v[I,0]*(2.0*rh1[2,0]+3.0*rh1[1,0]-6.0*rh1[0,0]+rh1[Nx-1,0])/(6.0*dx)
            Frho1[I,0] =v[I,0]*(2.0*rh1[1,0]+3.0*rh1[0,0]-6.0*rh1[Nx-1,0]+rh1[Nx-2,0])/(6.0*dx)
#                Frho1[I,J] = v[I,0]*(3.0*rh1[0,J]-4.0*rh1[0,J-1]+rh1[0,J-2])/(2.0*dx)
#            Frho1[I,1] = v[I,0]*(3.0*rh1[0,1]-4.0*rh1[0,0]+rh1[0,Nx-1])/(2.0*dx)
#            Frho1[I,0] =v[I,0]*(3.0*rh1[0,0]-4.0*rh1[0,Nx-1]+rh1[0,Nx-2])/(2.0*dx)    

#    for I in range(0,Nv):
#        if v[I]>0:
#            for J in range(1,Nx-1):
#                Frho1[I,J] = -v[I,0]*(1.0*rh1[0,J+1]-1.0*rh1[0,J-1])/(2.0*dx)
#            Frho1[I,0] = -v[I,0]*(1.0*rh1[0,1]-1.0*rh1[0,Nx-1])/(2.0*dx)
#            Frho1[I,Nx-1] = -v[I,0]*(1.0*rh1[0,0]-1.0*rh1[0,Nx-2])/(2.0*dx)
#        else:
#            for J in range(1,Nx-1):
#                Frho1[I,J] = v[I,0]*(1.0*rh1[0,J+1]-1.0*rh1[0,J-1])/(2.0*dx)
#            Frho1[I,0] = v[I,0]*(1.0*rh1[0,1]-1.0*rh1[0,Nx-1])/(2.0*dx)
#            Frho1[I,Nx-1] = v[I,0]*(1.0*rh1[0,0]-1.0*rh1[0,Nx-2])/(2.0*dx)
            

    for I in range(0,Nv):
        for J in range(0,Nx):
            gk1test[I,J,tt]=rh1[J,0]
#            gk1test[I,J,tt]=gk1[I,J]
#    gk1test[:,:,tt]=rh1[:,:]
    UFgk1test[:,:,tt]=Frho1[:,:]
#    UFgk1test[:,:,tt]=UFgk1[:,:]   
    
    
    
    gk2=(eps**2*g-dt*A1[1,0]*(UFgk1-PUFgk1)*eps-dt*A1[1,0]*Frho1-dt*A1[1,0]*0.0*gk1*eps**2)/(eps**2+dt*A2[1,1]*sigmaS)
    
    
    
        
#    for I in range(0,Nv):
#        for J in range(0,Nx):
#            Data_testsave[I,J,tt]=gk2[I,J]
    
    

#    gLk2[:,0] = gk2[:,Nx-1]
#    gRk2[:,0] = gk2[:,0]
    

    for I in range(0,Nv):
        if v[I]>0:
            for J in range(2,Nx-1):
                UFgk2[I,J] =v[I,0]*(2.0*gk2[I,J+1]+3.0*gk2[I,J]-6.0*gk2[I,J-1]+gk2[I,J-2])/(6.0*dx)
            UFgk2[I,Nx-1] =v[I,0]*(2.0*gk2[I,0]+3.0*gk2[I,Nx-1]-6.0*gk2[I,Nx-2]+gk2[I,Nx-3])/(6.0*dx)
            UFgk2[I,1] = v[I,0]*(2.0*gk2[I,2]+3.0*gk2[I,1]-6.0*gk2[I,0]+gk2[I,Nx-1])/(6.0*dx)
            UFgk2[I,0] =v[I,0]*(2.0*gk2[I,1]+3.0*gk2[I,0]-6.0*gk2[I,Nx-1]+gk2[I,Nx-2])/(6.0*dx)          
        else:
            for J in range(1,Nx-2):
                UFgk2[I,J] =v[I,0]*(-1.0*gk2[I,J+2]+6.0*gk2[I,J+1]-3.0*gk2[I,J]-2.0*gk2[I,J-1])/(6.0*dx)
            UFgk2[I,0] =v[I,0]*(-1.0*gk2[I,2]+6.0*gk2[I,1]-3.0*gk2[I,0]-2.0*gk2[I,Nx-1])/(6.0*dx)
            UFgk2[I,Nx-2] =v[I,0]*(-1.0*gk2[I,0]+6.0*gk2[I,Nx-1]-3.0*gk2[I,Nx-2]-2.0*gk2[I,Nx-3])/(6.0*dx)
            UFgk2[I,Nx-1] =v[I,0]*(-1.0*gk2[I,1]+6.0*gk2[I,0]-3.0*gk2[I,Nx-1]-2.0*gk2[I,Nx-2])/(6.0*dx)


        
        
        
    for i in range(0,Nx):
        for j in range(0,Nv):
            s[j,0]=wv[j,0]*UFgk2[j,i]########Could this be a problem
        ssk2[i,0]=sum(s)/2
    for j in range(0,Nv):
        for i in range(0,Nx):
            PUFgk2[j,i] = ssk2[i,0] #pvdxg extended to all v
            
            
    
            
    rh2=rho-dt*A1[1,0]*(0.0*rh1-G)-dt*A2[1,1]*ssk2#PUFgk2


#    for I in range(0,Nv):
#        for J in range(0,Nx):
#            Data_testsave2[I,J,tt]=rh2[0,J]
    

        
    for I in range(0,Nv):        
        if v[I]>0:
#            for J in range(0,Nx-2):#
#                Frho2[I,J] = v[I,0]*(-1.0*rh2[0,J+2]+4.0*rh2[0,J+1]-3.0*rh2[0,J])/(2.0*dx)
#            Frho2[I,Nx-2] = v[I,0]*(-1.0*rh2[0,0]+4.0*rh2[0,Nx-1]-3.0*rh2[0,Nx-2])/(2.0*dx)
#            Frho2[I,Nx-1] =v[I,0]*(-1.0*rh2[0,1]+4.0*rh2[0,0]-3.0*rh2[0,Nx-1])/(2.0*dx)           
            for J in range(1,Nx-2):
                Frho2[I,J] =v[I,0]*(-1.0*rh2[J+2,0]+6.0*rh2[J+1,0]-3.0*rh2[J,0]-2.0*rh2[J-1,0])/(6.0*dx)
            Frho2[I,0] = v[I,0]*(-1.0*rh2[2,0]+6.0*rh2[1,0]-3.0*rh2[0,0]-2.0*rh2[Nx-1,0])/(6.0*dx)
            Frho2[I,Nx-2] =v[I,0]*(-1.0*rh2[0,0]+6.0*rh2[Nx-1,0]-3.0*rh2[Nx-2,0]-2.0*rh2[Nx-3,0])/(6.0*dx)
            Frho2[I,Nx-1] =v[I,0]*(-1.0*rh2[1,0]+6.0*rh2[0,0]-3.0*rh2[Nx-1,0]-2.0*rh2[Nx-2,0])/(6.0*dx)    
        else:
            for J in range(2,Nx-1):
                Frho2[I,J] =v[I,0]*(2.0*rh2[J+1,0]+3.0*rh2[J,0]-6.0*rh2[J-1,0]+rh2[J-2,0])/(6.0*dx)
            Frho2[I,Nx-1] =v[I,0]*(2.0*rh2[0,0]+3.0*rh2[Nx-1,0]-6.0*rh2[Nx-2,0]+rh2[Nx-3,0])/(6.0*dx)
            Frho2[I,1] = v[I,0]*(2.0*rh2[2,0]+3.0*rh2[1,0]-6.0*rh2[0,0]+rh2[Nx-1,0])/(6.0*dx)
            Frho2[I,0] =v[I,0]*(2.0*rh2[1,0]+3.0*rh2[0,0]-6.0*rh2[Nx-1,0]+rh2[Nx-2,0])/(6.0*dx)
#            for J in range(2,Nx):#
#                Frho2[I,J] = v[I,0]*(3.0*rh2[0,J]-4.0*rh2[0,J-1]+rh2[0,J-2])/(2.0*dx)
#            Frho2[I,1] = v[I,0]*(3.0*rh2[0,1]-4.0*rh2[0,0]+rh2[0,Nx-1])/(2.0*dx)#
#            Frho2[I,0] =v[I,0]*(3.0*rh2[0,0]-4.0*rh2[0,Nx-1]+rh2[0,Nx-2])/(2.0*dx)
            
            
            
#    for I in range(0,Nv):
#        if v[I]>0:
#            for J in range(1,Nx-1):
#                Frho2[I,J] = -v[I,0]*(1.0*rh2[0,J+1]-1.0*rh2[0,J-1])/(2.0*dx)
#            Frho2[I,0] = -v[I,0]*(1.0*rh2[0,1]-1.0*rh2[0,Nx-1])/(2.0*dx)#
#            Frho2[I,Nx-1] = -v[I,0]*(1.0*rh2[0,0]-1.0*rh2[0,Nx-2])/(2.0*dx)
#        else:
#            for J in range(1,Nx-1):
#                Frho2[I,J] = v[I,0]*(1.0*rh2[0,J+1]-1.0*rh2[0,J-1])/(2.0*dx)
#            Frho2[I,0] = v[I,0]*(1.0*rh2[0,1]-1.0*rh2[0,Nx-1])/(2.0*dx)
#            Frho2[I,Nx-1] = v[I,0]*(1.0*rh2[0,0]-1.0*rh2[0,Nx-2])/(2.0*dx)
            
            
    
    gk3=(eps**2*g-dt*A1[2,0]*(UFgk1-PUFgk1)*eps-dt*A1[2,0]*Frho1-dt*A1[2,1]*eps*(UFgk2-PUFgk2)-dt*A1[2,1]*Frho2-dt*A1[2,0]*gk1*0.0*eps**2-dt*A1[2,1]*0.0*gk2*eps**2-dt*A2[2,1]*1.0*gk2)/(eps**2+dt*A2[2,2]*sigmaS)
    
    
    
    for I in range(0,Nv):
        for J in range(0,Nx):
            Data_testsave[I,J,tt]=gk3[I,J]
    
    

    for I in range(0,Nv):
        if v[I]>0:
            for J in range(2,Nx-1):
                UFgk3[I,J] =v[I,0]*(2.0*gk3[I,J+1]+3.0*gk3[I,J]-6.0*gk3[I,J-1]+gk3[I,J-2])/(6.0*dx)
            UFgk3[I,Nx-1] =v[I,0]*(2.0*gk3[I,0]+3.0*gk3[I,Nx-1]-6.0*gk3[I,Nx-2]+gk3[I,Nx-3])/(6.0*dx)
            UFgk3[I,1] = v[I,0]*(2.0*gk3[I,2]+3.0*gk3[I,1]-6.0*gk3[I,0]+gk3[I,Nx-1])/(6.0*dx)
            UFgk3[I,0] =v[I,0]*(2.0*gk3[I,1]+3.0*gk3[I,0]-6.0*gk3[I,Nx-1]+gk3[I,Nx-2])/(6.0*dx)          
        else:
            for J in range(1,Nx-2):
                UFgk3[I,J] =v[I,0]*(-1.0*gk3[I,J+2]+6.0*gk3[I,J+1]-3.0*gk3[I,J]-2.0*gk3[I,J-1])/(6.0*dx)
            UFgk3[I,0] =v[I,0]*(-1.0*gk3[I,2]+6.0*gk3[I,1]-3.0*gk3[I,0]-2.0*gk3[I,Nx-1])/(6.0*dx)
            UFgk3[I,Nx-2] =v[I,0]*(-1.0*gk3[I,0]+6.0*gk3[I,Nx-1]-3.0*gk3[I,Nx-2]-2.0*gk3[I,Nx-3])/(6.0*dx)
            UFgk3[I,Nx-1] =v[I,0]*(-1.0*gk3[I,1]+6.0*gk3[I,0]-3.0*gk3[I,Nx-1]-2.0*gk3[I,Nx-2])/(6.0*dx)        
        
    for i in range(0,Nx):
        for j in range(0,Nv):
            s[j,0]=wv[j,0]*UFgk3[j,i]########Could this be a problem
        #ss[i,0]=sum(s)/2#pvdxg, unextended to all v
        ssk3[i,0]=sum(s)/2
    for j in range(0,Nv):
        for i in range(0,Nx):
            PUFgk3[j,i] = ssk3[i,0] #pvdxg extended to all v

    rh3=rho-dt*A1[2,0]*(0.0*rh1-G)-dt*A1[2,1]*(0.0*rh2-G)-dt*A2[2,1]*ssk2-dt*A2[2,2]*ssk3
    
    
    for I in range(0,Nv):
        for J in range(0,Nx):
            Data_testsave2[I,J,tt]=rh3[J,0]
    
    
    g1=g-dt*w1[0,0]*((UFgk1-PUFgk1)/eps+Frho1/eps**2.0+0.0*gk1)-dt*w1[1,0]*((UFgk2-PUFgk2)/eps+Frho2/eps**2.0+0.0*gk2)-dt*w2[1,0]*(sigmaS/eps**2.0)*gk2-dt*w2[2,0]*(sigmaS/eps**2.0)*gk3
    

    rho1=rho-dt*w1[0,0]*(0.0*rh1-G)-dt*w1[1,0]*(0.0*rh2-G)-dt*w2[1,0]*ssk2-dt*w2[2,0]*ssk3
    
    rho1=rho1
    g = g1
    




#    UFg[:,:] = (np.roll(g,-1,axis=1)-np.roll(g,1,axis=1))/(2.0*dx)
#    for j in range(0,Nv):
#        UFg[j,:] = v[j,0]*UFg[j,:]
        
        
        
        
                
    for I in range(0,Nv):
        if v[I]>0:
            for J in range(2,Nx):
                UFg[I,J] =v[I,0]*(3.0*g[I,J]-4.0*g[I,J-1]+g[I,J-2])/(2.0*dx)
            UFg[I,1] = v[I,0]*(3.0*g[I,1]-4.0*g[I,0]+g[I,Nx-1])/(2.0*dx)
            UFg[I,0] =v[I,0]*(3.0*g[I,0]-4.0*g[I,Nx-1]+g[I,Nx-2])/(2.0*dx) 
        else:
            for J in range(0,Nx-2):
                UFg[I,J] = v[I,0]*(-1.0*g[I,J+2]+4.0*g[I,J+1]-3.0*g[I,J])/(2.0*dx)
            UFg[I,Nx-2] = v[I,0]*(-1.0*g[I,0]+4.0*g[I,Nx-1]-3.0*g[I,Nx-2])/(2.0*dx)
            UFg[I,Nx-1] =v[I,0]*(-1.0*g[I,1]+4.0*g[I,0]-3.0*g[I,Nx-1])/(2.0*dx)
        
        
        
        
    for i in range(0,Nx):
        for j in range(0,Nv):
            s[j,0]=wv[j,0]*UFg[j,i]########Could this be a problem
        ss[i,0]=sum(s)/2#pvdxg, unextended to all v
    for j in range(0,Nv):
        for i in range(0,Nx):
            PUFg[j,i] = ss[i,0] #pvdxg extended to all v
    
#    Frho[:,:] = (np.roll(rho1,-1,axis=1)-np.roll(rho1,1,axis=1))/(2.0*dx)
#    for j in range(0,Nv):
#        Frho[j,:] = v[j,0]*Frho[j,:]
        
        
        
        
        
        
    for I in range(0,Nv):
        if v[I]>0:
            for J in range(0,Nx-2):#
                Frho[I,J] = v[I,0]*(-1.0*rho1[J+2,0]+4.0*rho1[J+1,0]-3.0*rho1[J,0])/(2.0*dx)
            Frho[I,Nx-2] = v[I,0]*(-1.0*rho1[0,0]+4.0*rho1[Nx-1,0]-3.0*rho1[Nx-2,0])/(2.0*dx)
            Frho[I,Nx-1] =v[I,0]*(-1.0*rho1[1,0]+4.0*rho1[0,0]-3.0*rho1[Nx-1,0])/(2.0*dx)
        else:
            for J in range(2,Nx):#
                Frho[I,J] = v[I,0]*(3.0*rho1[J,0]-4.0*rho1[J-1,0]+rho1[J-2,0])/(2.0*dx)
            Frho[I,1] = v[I,0]*(3.0*rho1[1,0]-4.0*rho1[0,0]+rho1[Nx-1,0])/(2.0*dx)#
            Frho[I,0] =v[I,0]*(3.0*rho1[0,0]-4.0*rho1[Nx-1,0]+rho1[Nx-2,0])/(2.0*dx)
        
        
        




    pvdxg_data[:,tt]=ss[0,:]

    Data_idp[:,:,tt]=UFg-PUFg

    Data_Frho[:,:,tt]=Frho[:,:]
    
#    Data_g[:,:,tt]=g[:,:]
    rho = rho1
#    Data_rho[:,tt]=rho[0,:]
  #  tt
  #  nt
  
print('1')
fig = plt.figure()
plt.scatter(x,rho, color='black', linewidth=3, label='Spectral solution')
plt.legend()
plt.show() 

print('2')
n1=1
n2=1
NNN=1
flag=1
if flag==1:
    if n1>n2:
#        seven=NNN**2
        seven=8
        NNN=1
    else:
#        seven=NNN
        seven=8
        NNN=1
    Data_rho2 = np.zeros([int(Nx/NNN),int(nt/seven)],dtype=float)
    Data_g2 = np.zeros([Nv,int(Nx/NNN),int(nt/seven)],dtype=float)
    
    for I in range(0,Nv):
        for J in range(0,int(Nx/NNN)):
            for K in range(0,int(nt/seven)):
                Data_g2[I,J,K]=Data_g[I,int(J*NNN),int(K*seven)]
               # Data_g2[I,J,K]=Data_testsave[I,int(J*NNN),int(K*8.0)]
                
    for J in range(0,int(Nx/NNN)):
        for K in range(0,int(nt/seven)):
            Data_rho2[J,K]=Data_rho[int(NNN*J),int(K*seven)]

    dt=dt*seven
    dx=dx*NNN
    savemat('data_mm_periodic64c.mat',{'x':x, 'u':Data_rho2,'g':Data_g2 , 'v':v, 'wv':wv, 'dt':dt, 'vrho_x':Data_Frho, 'vg_x-pvg_x':Data_idp, 'pvdxg_data':pvdxg_data,'hx':dx ,'eps':eps })

    savemat('datatest.mat',{'test':Data_testsave,'test2':Data_testsave2})
else:
    Data_rho2 = np.zeros([Nv,int(Nx),int(nt)],dtype=float)
    Data_g2 = np.zeros([Nv,int(Nx),int(nt)],dtype=float)
    for I in range(0,Nv):
        for J in range(0,int(Nx)):
            for K in range(0,int(nt)):
                #            Data_g2[I,J,K]=Data_g[I,int(J*NNN),int(K*8.0)]
                Data_g2[I,J,K]=Data_testsave[I,J,K]
                
    for I in range(0,Nv):
        for J in range(0,Nx):
            for K in range(0,int(nt)):
                Data_rho2[I,J,K]=Data_testsave2[I,J,K]

    dt=dt
    dx=dx
    
#    savemat('data_mm_periodic.mat',{'x':x, 'u':Data_rho,'g':Data_g , 'v':v, 'wv':wv, 'dt':dt, 'vrho_x':Data_Frho, 'vg_x-pvg_x':Data_idp, 'pvdxg_data':pvdxg_data,'hx':dx })
    savemat('datatest.mat',{'test':Data_testsave,'test2':Data_testsave2})
    savemat('data_mm_periodic64c.mat',{'x':x, 'u':Data_rho2,'g':Data_g2 , 'v':v, 'wv':wv, 'dt':dt, 'vrho_x':Data_Frho, 'vg_x-pvg_x':Data_idp, 'pvdxg_data':pvdxg_data,'hx':dx, 'eps':eps })
    
#Nm=50
#saverho2=np.zeros([1,Nm],dtype=float)
#for I in range(0,Nm):
#    saverho2[0,I]=rho[0,I*1]

#saverho64=rho


#savemat('data_mm_periodic.mat',{'x':x, 'u':Data_rho,'g':Data_g , 'v':v, 'wv':wv, 'dt':dt, 'vrho_x':Data_Frho, 'vg_x-pvg_x':Data_idp, 'pvdxg_data':pvdxg_data })

#savemat('datatest.mat',{'test':Data_testsave,'test2':Data_testsave2})

#SavedData = scipy.io.loadmat('data_heat.mat')
    
test=np.zeros([Nv,Nx],dtype=float)
for I in range(0,Nv):
    for J in range(0,Nx):
        test[I,J]=wv[I,0]*g[I,J]
cond=np.sum(test,0)
print(np.min(cond))
print(np.max(cond))


test=np.zeros([Nv,int(Nx/NNN)],dtype=float)
for I in range(0,Nv):
    for J in range(0,int(Nx/NNN)):
        test[I,J]=wv[I,0]*Data_g2[I,J,K]
cond=np.sum(test,0)
print(np.min(cond))
print(np.max(cond))


