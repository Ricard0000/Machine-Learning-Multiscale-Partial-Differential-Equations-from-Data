#Production of data for linear kinetic equation (its a coupled system of \rho and g):

#\partial_{t}\rho&=-\partial_{x}\langle vg\rangle-\sigma^{A}\rho + G
#\partial_{t}g&=-\dfrac{1}{\veps}\left(\mathcal{I}-\langle \ \rangle\right)(v\partial_{x}g)-\dfrac{1}{\veps^{2}}v\partial_{x}\rho-\dfrac{\sigma^{S}}{\veps^{2}}g-\sigma^{A}g,


import sys
sys.path.insert(0, '../../Utilities/')
import scipy.io
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt



#For 1/64
#Nx=1600
#dt=dt=(2.5*10**-7)


Lx=1.0
Nx=100
eps=1.0/32.0
#eps=1.0/65536.0
Nv=16

a=0.0

dx = Lx/(Nx)
b=Lx-dx
x=np.zeros([Nx,1],dtype=float)
for I in range(0,Nx):
    x[I,0] = dx*(I)

    
TTdata = scipy.io.loadmat('Periodic_data.mat')
v = TTdata['v']
wv=TTdata['wv']

rho = np.zeros([1,Nx],dtype=float)
for I in range(0,Nx):
#    rho[0,I] =1*np.exp(-50.0*(x[I,0]-Lx/2)**2.0)#+1*np.exp(-1000.0*(x[I,0]-2.5+1)**2.0)
#    rho[0,I] = 1+np.exp(x[I,0])*np.sin(1.0*np.pi*x[I,0])
    rho[0,I] = 1+np.sin(2.0*np.pi*x[I,0])
#    rho[0,I] =20*x[I,0]*(x[I,0])*(x[I,0]-1.5)*np.sin(2.0*np.pi*x[I,0])**2*np.exp(-15.0*(x[I,0]-0.75)**2.0)
#    rho[0,I] =(x[I,0]-0.5)**1*np.exp(-256.0*(x[I,0]-0.5)**2.0)#+1*np.exp(-1000.0*(x[I,0]-2.5+1)**2.0)   
#    rho[0,I] = 1+np.sin(2.0*np.pi*x[I,0])#*np.exp(-300.0*(x[I,0])**2.0)
#rho[0,I]=rho[0,0]    
g = np.zeros([Nv,Nx],dtype=float)

sigmaA = np.zeros([1,Nx],dtype=float)
sigmaAg = np.zeros([Nx],dtype=float)
#sigmaArho = np.zeros([1,Nx+1],dtype=float)

sigmaS = np.ones([Nx],dtype=float)

G = np.zeros([1,Nx],dtype=float)

#fL = v
#fR = np.zeros([Nv,1],dtype=float)

UFg = np.zeros([Nv,Nx],dtype=float)

#PUFg = np.zeros([1,Nx],dtype=float)
PUFg = np.zeros([Nv,Nx],dtype=float)
#PUFg = np.zeros([1,Nx],dtype=float)
Frho = np.zeros([Nv,Nx],dtype=float)
CFg = np.zeros([Nv,Nx],dtype=float)
PCFg = np.zeros([1,Nx],dtype=float)

gL = np.zeros([Nv,1],dtype=float)
gR = np.zeros([Nv,1],dtype=float)


dt = (3*np.min(sigmaS)*dx**2/2+eps*dx)/(2.0*2)

#dt=(5.0*10**-6)
#dt=(2.5*10**-8)

#dt=dx*dx/200
dt=0.000000005

#nt=(10000-8*250)*4
nt=32*64
tmax=dt*nt

CFL=dt/dx

timeskip=1#250



dtp=dt*timeskip
dxp=dtp/CFL

sugss=dxp/dx


#dt=np.float32(tmax/(timeskip*2.0**4))
#dt = (3*np.min(sigmaS)*dx**2/2+eps*dx)/(2.0*2)

#nt = round(tmax/dt)

Data_rho = np.zeros([Nx,int(nt/timeskip)],dtype=float)
Data_g = np.zeros([Nv,Nx,int(nt/timeskip)],dtype=float)

s = np.zeros([Nv,1],dtype=float)
ss = np.zeros([Nx,1],dtype=float)

vplus = np.zeros([Nv,1],dtype=float)
vminus = np.zeros([Nv,1],dtype=float)



sigmaS=np.zeros([Nx,1],dtype=float)
for I in range(0,Nx):
#    sigmaS[I,0]=1.0/3.0#+1.0*x[I,0]-1.0*x[I,0]**2.0
    sigmaS[I,0]=1.0#(6+40*(x[I,0]-0.5)*np.exp(-500.0*(x[I,0]-Lx/2)**4.0))
L=-1
for tt in range(0,int(nt)):

    print(tt/nt)
    gL[:,0] = g[:,Nx-1]
    gR[:,0] = g[:,0]
    
#######################################################################
#    T H I S    I S   T H E   A D V E C T I O N    O P E R A T O R    #
#######################################################################
    for j in range(0,Nv):
        if v[j]>0:
            UFg[j,1:Nx] = v[j,0]*(g[j,1:Nx]-g[j,0:Nx-1])/dx
            UFg[j,0] = v[j,0]*(g[j,0]-gL[j])/dx
        else:
            UFg[j,0:(Nx-1)] = v[j,0]*(g[j,1:Nx]-g[j,0:(Nx-1)])/dx
            UFg[j,Nx-1] = v[j,0]*(gR[j]-g[j,Nx-1])/dx
########################################################################
#                                                                      #
######################################################################## 
                         
#########################################################################
#    T H I S    I S   T H E   P R O J E C T I O N    O P E R A T O R    #
#########################################################################
    for i in range(0,Nx):  ####Projection of UFg
        for j in range(0,Nv):
            s[j,0]=wv[j,0]*UFg[j,i]#
        ss[i,0]=sum(s)/2#
##########################################################################        
#                                                                        #
##########################################################################        
    for j in range(0,Nv):          #This extends
        for i in range(0,Nx):      #to (v,x)
            PUFg[j,i] = ss[i,0]    #space
##########################################################################        
#                                                                        #
##########################################################################

    rhoN = rho[0,0]
    for j in range(0,Nv):
        Frho[j,0:Nx-1] = v[j,0]*(rho[0,1:Nx]-rho[0,0:Nx-1])/dx
        Frho[j,Nx-1] = v[j,0]*(rhoN-rho[0,Nx-1])/dx

    gtmp = eps**2*g - dt*eps*(UFg-PUFg)
    g1 = gtmp - dt*Frho - dt*eps**2*g*0.0


#    gtmp = - dt*eps*(UFg-PUFg)
#    g1 = gtmp - dt*Frho - dt*eps**2*g*0.0

#    divide=(eps**2+dt*1.0)

    for I in range(0,Nv):
        for J in range(0,Nx):
            g1[I,J]=g1[I,J]/(eps**2+dt*sigmaS[J,0])

#for J in range(0,Nx):
#gk2[I,J]=(eps**2*g[I,J]-dt*A1[1,0]*(UFgk1[I,J]-PUFgk1[I,J])*eps-dt*A1[1,0]*Frho1[I,J]-dt*A1[1,0]*0.0*gk1[I,J]*eps**2)/(eps**2+dt*A2[1,1]*sigmaS[J,0])
    
#    g1 = g1/divide
    g1L = g1[:,Nx-1]
    
    for j in range(0,Nv):
        CFg[j,1:Nx] = v[j,0]*(g1[j,1:Nx]-g1[j,0:(Nx-1)])/dx
        CFg[j,0] = v[j]*(g1[j,0]-g1L[j])/dx#for periodic
    PCFg = sum(wv*CFg)/2#

    rho1 = rho - dt*PCFg + dt*G

    rho = rho1
    g = g1   

    if tt % timeskip == 0:
        L=L+1
        print(L)
        Data_g[:,:,L]=g[:,:]
        Data_rho[:,L]=rho[0,:]

spaceskip=1#int(sugss)
#spaceskip=4.0
x2=np.zeros([int(Nx/spaceskip),1])
Data_g2=np.zeros([Nv,int(Nx/spaceskip),int(nt/timeskip)])
Data_rho2=np.zeros([int(Nx/spaceskip),int(nt/timeskip)])
for I in range(0,Nv):
    for J in range(0,int(Nx/spaceskip)):
        for K in range(0,int(nt/timeskip)):
            Data_g2[I,J,K]=Data_g[I,int(spaceskip*J),K]
            
for J in range(0,int(Nx/spaceskip)):
    for K in range(0,int(nt/timeskip)):
        Data_rho2[J,K]=Data_rho[int(spaceskip*J),K]
for I in range(0,int(Nx/spaceskip)):
     x2[I,0]=x[int(spaceskip*I),0]
fig = plt.figure()
plt.scatter(x,rho, color='black', linewidth=3, label='Rho at final time')
plt.legend()
plt.show()

#########savemat('data_mm_periodic.mat',{'x':x, 'u':Data_rho,'g':Data_g , 'v':v, 'wv':wv, 'dt':timeskip*dt, 'vrho_x':Data_Frho, 'vg_x-pvg_x':Data_idp, 'pvdxg_data':pvdxg_data,'hx':dx })
dt=dt*timeskip
#savemat('data_mm_periodic.mat',{'x':x, 'u':Data_rho,'g':Data_g , 'v':v, 'wv':wv, 'dt':dt,'hx':dx })
dx=dx*spaceskip
savemat('data_mm_periodic32z.mat',{'x':x2,'u':Data_rho2,'g':Data_g2 , 'v':v, 'wv':wv, 'dt':dt,'hx':dx, 'a':a, 'b':b, 'eps':eps})




vv=np.ones([Nv],dtype=float)
xx=np.ones([Nx],dtype=float)
for I in range(0,Nv):
    vv[I]=I
for I in range(0,Nx):
    xx[I]=I

vvv,xxx=np.meshgrid(xx,vv)


#temp=Data_g[:,:,0]
#fig = plt.figure()
#ax2 = plt.axes(projection='3d')
#ax2.plot_wireframe(vvv,xxx, temp, color='r')


dgx=(np.roll(Data_g,1,axis=1)-np.roll(Data_g,-1,axis=1))/(2*dx)

#fig = plt.figure()
#ax2 = plt.axes(projection='3d')
#ax2.plot_wireframe(vvv,xxx, dgx[:,:,0], color='r')




xx2=np.ones([int(Nx/spaceskip)],dtype=float)
for I in range(0,int(Nx/spaceskip)):
    xx2[I]=I

vvv,xxx2=np.meshgrid(xx2,vv)
temp=Data_g2[:,:,10]
#fig = plt.figure()
#ax2 = plt.axes(projection='3d')
#ax2.plot_wireframe(vvv,xxx2, temp, color='r')











gx=(np.roll(Data_g2,1,axis=1)-np.roll(Data_g2,-1,axis=1))/(2*dx)



temp=gx[:,:,19]
#fig = plt.figure()
#ax2 = plt.axes(projection='3d')
#ax2.plot_wireframe(vvv,xxx2, temp, color='r')


a=0.0
b=Lx       
kkk=np.zeros([1,int(Nx/spaceskip)],dtype=float)
Nx_half=round(int(Nx/spaceskip)/2)
for I in range(0,Nx_half+1):
    kkk[0,I]=2.0*np.pi/(b-a+dx)*I
 
for I in range(0,Nx_half-1): 
    kkk[0,int(Nx/spaceskip)-I-1]=-2.0*np.pi/(b-a+dx)*(I+1)

gx2=np.zeros([Nv,int(Nx/spaceskip),int(nt/spaceskip)])
for I in range(0,Nv):
    for K in range(0,int(nt/timeskip)):
        gx2[I,:,K]=np.real(np.fft.ifft(-1j*kkk*np.fft.fft(Data_g2[I,:,K])))

temp=gx[:,:,19]-gx2[:,:,19]
#fig = plt.figure()
#ax2 = plt.axes(projection='3d')
#ax2.plot_wireframe(vvv,xxx2, temp, color='r')


















dx=dx/spaceskip
gx=(np.roll(Data_g,1,axis=1)-np.roll(Data_g,-1,axis=1))/(2*dx)



vv=np.ones([Nv],dtype=float)
xx=np.ones([Nx],dtype=float)
for I in range(0,Nv):
    vv[I]=I
for I in range(0,Nx):
    xx[I]=I

vvv,xxx=np.meshgrid(xx,vv)


temp=gx[:,:,19]
fig = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.plot_wireframe(vvv,xxx, temp, color='r')


a=0.0
b=Lx-dx
kkk=np.zeros([1,Nx],dtype=float)
Nx_half=round(int(Nx/2))
for I in range(0,Nx_half+1):
    kkk[0,I]=2.0*np.pi/(b-a+dx)*I
 
for I in range(0,Nx_half-1): 
    kkk[0,Nx-I-1]=-2.0*np.pi/(b-a+dx)*(I+1)

gx2=np.zeros([Nv,Nx,int(nt/timeskip)])
for I in range(0,Nv):
    for K in range(0,int(nt/timeskip)):
        gx2[I,:,K]=np.real(np.fft.ifft(-1j*kkk*np.fft.fft(Data_g[I,:,K])))




temp=gx2[:,:,19]
fig = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.plot_wireframe(vvv,xxx, temp, color='r')






#temp=gx[:,:,19]-gx2[:,:,19]
#fig = plt.figure()
#ax2 = plt.axes(projection='3d')
#ax2.plot_wireframe(vvv,xxx, temp, color='r')









