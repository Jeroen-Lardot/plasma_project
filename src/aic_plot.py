import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from IPython.display import Image
import scipy.io
import pandas as pd
from IPython.display import display, clear_output
import sys, os
import warnings
import h5py
#from fcmeans import FCM
warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from scipy import linalg
import itertools
#get_ipython().run_line_magic('matplotlib', 'inline')
from aidapy import load_data
import aidapy.aidaxr
from scipy.interpolate import griddata
from sklearn import mixture
from scipy.ndimage import gaussian_filter

### parameters inizialization ###

#set time
start_time = datetime(2017, 10, 16, 13, 7, 0);
end_time   = datetime(2017, 10, 16, 13, 8, 0);

#choose mms probe
probes= 3

#various
vmax = 600

#dimension of interpolation grid
grid_dim=50

#h5py file name
filename = 'outto_tst.h5'

#max components for gmm
n_components_range= 15

#number of generated particles from vdf
n_part=40000

#choose between bic or aic
information_criterion='bic'



### write vtk file for paraview ###
def write_vtk(ds_arr,x,y,z,itime):
    [nx, ny, nz] = np.shape(ds_arr)
    a=nx*ny*nz
    f = open('../'+str(information_criterion)+'/VTK_data'+str(itime)+'.vtk','w') # change your vtk file name
    f.write('# vtk DataFile Version 2.0\n')
    f.write('test\n')
    f.write('ASCII\n')
    f.write('DATASET STRUCTURED_POINTS\n')
    f.write('DIMENSIONS '+str(nx)+' '+str(ny)+' '+str( nz)+'\n') # change your dimension
    f.write('SPACING '+ str(x[1,0,0]-x[0,0,0]) +' ' + str(y[0,1,0]-y[0,0,0]) +' ' + str(z[0,0,1]-z[0,0,0]) +'\n')
    f.write('ORIGIN '+ str(x[0,0,0]) +' ' + str(y[0,0,0]) +' ' + str(z[0,0,0]) +'\n')
    f.write('POINT_DATA '+str(nx*ny*nz)+'\n') # change the number of point data
    f.write('SCALARS VDF float\n')
    f.write('LOOKUP_TABLE default\n')

    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz):
                content = str(ds_arr[i,j,k])
                f.write(content+'\n')
    f.close()
    return


### download data  ###
settings = {'prod': ['i_dist', 'sc_pos', 'sc_att'],
            'probes': probes, 'coords': 'gse', 'mode': 'high_res', 'frame':'gse'}

xr_mms = load_data(mission='mms', start_time=start_time, end_time=end_time, **settings)
print(xr_mms)
#physical quantities
mi = 1.67e-27
e = 0.5*mi*(vmax*1e3)**2
e = e/(1.6e-19)

#utility strings
phistr='mms{}_dis_phi_brst'.format(probes)
thetastr='mms{}_dis_theta_brst'.format(probes)
energystr='mms{}_dis_energy_brst'.format(probes)
fpistr='i_dist{}'.format(probes)

### cartesian fpi ###

fpi=np.array(xr_mms[fpistr])
phi=np.array(xr_mms[phistr]/360*2*np.pi)
theta=np.array(xr_mms[thetastr]/360*2*np.pi)
energy=np.array(xr_mms[energystr])

tt=np.array(xr_mms['time1']).astype('<i8')

v =np.sqrt(2/mi*energy*(1.6e-19)) *1e-3

fiv, thv, vv = np.meshgrid(phi, theta, v,  indexing='ij')
vxv = vv * np.cos (fiv) * np.sin (thv)
vyv = vv * np.sin (fiv) * np.sin (thv)
vzv = vv * np.cos (thv)

# print('vxv',np.min(vxv),np.max(vxv))
# print('vyv',np.min(vyv),np.max(vyv))
# print('vzv',np.min(vzv),np.max(vzv))

points=np.column_stack((vxv.ravel(), vyv.ravel(), vzv.ravel()))

vx = np.linspace(-vmax, vmax, grid_dim)

grid_x, grid_y, grid_z= np.meshgrid(vx,vx,vx, indexing='ij')

Nx,Ny,Nz= grid_x.shape
Ntimes=fpi.shape[0]
fpicart=np.zeros((Ntimes,Nx,Ny,Nz))

for itime in range(0, Ntimes):
    fpi1=fpi[itime,:].ravel()
    fpicart[itime,:,:,:] = griddata(points, fpi1, (grid_x, grid_y, grid_z), method='linear')
    
    ### writing vtk file with distributions for paraview ###
    write_vtk(fpicart[itime,:,:,:],grid_x, grid_y, grid_z,itime)

print(Ntimes)

### write h5py file with data and cartesian interpolation ###
file = h5py.File(filename,'w')
size = fpi.shape
file.create_dataset('fpi',size,data=fpi)
size = phi.shape
file.create_dataset('phi',size,data=phi)
size = theta.shape
file.create_dataset('theta',size,data=theta)
size = energy.shape
file.create_dataset('energy',size,data=energy)
size = grid_x.shape
file.create_dataset('grid_x',size,data=grid_x)
size = grid_y.shape
file.create_dataset('grid_y',size,data=grid_y)
size = grid_z.shape
file.create_dataset('grid_z',size,data=grid_z)
size = fpicart.shape
file.create_dataset('fpcart',size,data=fpicart)
size = tt.shape
file.create_dataset('times',size,'<i8',data=tt)
file.close()

### various vdf plots ###

### plot integrating over 1 axis ###
for i in range (0,Ntimes,10):
    fcut=fpicart[i,:,:,:]
    ftot1=np.sum(fcut, axis=0)
    ftot2=np.sum(fcut, axis=1)
    ftot3=np.sum(fcut, axis=2)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('fpi '+str(i)+' plan: xy,zx,yz')
    ax1.imshow(ftot1, cmap='jet')
    ax2.imshow(ftot2, cmap='jet')
    ax3.imshow(ftot3, cmap='jet')
    #plt.colorbar(scat)


nx,ny,nz=fpicart.shape[1],fpicart.shape[2],fpicart.shape[3]
nclusters_plot=[]
info_plot=[]

### particle generation from vdf ###
for i in range (0, Ntimes):
    vdf=fpicart[i,:,:,:]
    vmax=600
    dv= 2*vmax/(nx-1)
    v1D= np.arange(-vmax,vmax+dv,dv)
    vvx,vvy,vvz= np.meshgrid(v1D,v1D,v1D)
    Np=n_part

    f1D=vdf.flatten()
    fmin= min([j for j in f1D if j>0])

    f1D= np.where(f1D==0, fmin/1000, f1D)

    vx1D=(np.conjugate(vvx).T).flatten()
    vy1D=(np.conjugate(vvy).T).flatten()
    vz1D=(np.conjugate(vvz).T).flatten()

    np.random.seed(0)
    Ng=f1D.shape[0]
    ranarr=np.random.rand(4,Np)
    fcum=np.cumsum(f1D);

    fcum=Ng*fcum/fcum[Ng-1]

    NgRange=np.arange(1,Ng+1)

    Pg=np.interp(Ng*ranarr[0,:], fcum.T, NgRange)
    Pg= 1 + np.floor(Pg)
    Pg=Pg.astype(int)

    xp=vx1D[Pg] + dv*ranarr[1,:] - dv/2
    yp=vy1D[Pg] + dv*ranarr[2,:] - dv/2
    zp=vz1D[Pg] + dv*ranarr[3,:] - dv/2
    
    #store data for gmm 
    gmmdata=np.array([xp,yp,zp])
    gmmdata=np.conjugate(gmmdata).T

    ### gmm ###
    lowest_info_crit = np.infty
    info_crit= []
    
    for n_components in range (1, n_components_range):

        gmm = GaussianMixture(n_components,covariance_type='full' ,reg_covar=0.1, init_params='kmeans', random_state=0).fit(gmmdata)
        if (information_criterion=='aic'): info_crit.append(gmm.aic(gmmdata)) 
        elif (information_criterion=='bic'): info_crit.append(gmm.bic(gmmdata)) 
        
        if info_crit[-1] < lowest_info_crit:
                lowest_info_crit = info_crit[-1]
                best_gmm = gmm
    
    info_crit = np.array(info_crit)
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    clf = best_gmm
    bars = []


    fcm_labels = best_gmm.predict(gmmdata)
    nclusters_plot.append(best_gmm.n_components)
    info_plot.append(info_crit)
    print('probe:',probes, 'vdf:',i,'n_particles:',n_part,'info:',information_criterion,'gmm:',best_gmm.n_components, best_gmm.covariance_type)

plt.clf()        
plt.plot(nclusters_plot,'bo--')
plt.title('probe '+str(probes)+' n_particles '+str(n_part)+' info '+information_criterion)
plt.ylabel('gmm ecomponents')
plt.xlabel('time')
plt.show()

### plot aic/bic slope ###
cf=plt.imshow(np.transpose(info_plot)/np.amax(info_plot), origin = 'upper', extent=[0,itime,14,1], cmap='jet', aspect='auto')

plt.xlabel('time')
plt.ylabel('n of clusters')
# plt.colorbar(cf,format='%.10f')
plt.colorbar()



