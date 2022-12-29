from pyspedas.mms.particles.mms_part_slice2d import mms_part_slice2d
import pytplot
import pyspedas
import h5py
from scipy.interpolate import griddata
import numpy as np
import calendar
#time = '2016-01-21/01:06:50.799'
time = '2017-08-04/09:10:00.000'
vmax = 1e4
mi = 1.67e-27
me=mi/1836
e = 0.5*me*(vmax*1e3)**2
e = e/(1.6e-19)
 
#mms_part_slice2d(probe = '1', interpolation='2d', time=time, instrument='fpi', data_rate = 'brst', species='i', rotation='bv', erange=[0, e])


mms_dist=pyspedas.mms.fpi(probe = '3', trange=[time, time],datatype='des-dist', data_rate = 'brst')

print('Ended reading the data')

fpi=pytplot.data_quants['mms3_des_dist_brst'].values
print('fpi',fpi.shape)
phid=pytplot.data_quants['mms3_des_phi_brst']
print('phi',phid)
#phi=phid['data']/360*2*np.pi
phi=phid.values/360*2*np.pi
energy=pytplot.data_quants['mms3_des_energy_brst'].values
theta=pytplot.data_quants['mms3_des_theta_brst']['data']/360*2*np.pi
tt=pytplot.data_quants['mms3_des_dist_brst'].time
tt=tt.values
v =np.sqrt(2 / me * energy[0,:]*(1.6e-19)) *1e-3;
print('v',v.shape, phi.shape, theta.shape)
fiv, thv, vv = np.meshgrid(phi[0,:], theta, v,  indexing='ij')
vxv = vv * np.cos (fiv) * np.sin (thv)
vyv = vv * np.sin (fiv) * np.sin (thv)
vzv = vv * np.cos (thv)
print('vxv',np.min(vxv),np.max(vxv))
print('vyv',np.min(vyv),np.max(vyv))
print('vzv',np.min(vzv),np.max(vzv))

#print('vxv',vxv.shape)
#points=np.zeros((32*16*32,3))
#points[:,0]=vxv.ravel()
#points[:,1]=vyv.ravel()
#points[:,2]=vzv.ravel()
points=np.column_stack((vxv.ravel(), vyv.ravel(), vzv.ravel()))

#vxv=np.zeros((32,16,32))
#vyv=np.zeros((32,16,32))
#vzv=np.zeros((32,16,32))
#for i in range(0,32):
#   for j in range(0,16):
#       for k in range(0,32):
#          vxv[i,j,k] = v[i] * np.cos (phi[k]-np.pi) * np.sin (theta[j])
#          vyv[i,j,k] = v[i] * np.sin (phi[k]-np.pi) * np.sin (theta[j])
#          vzv[i,j,k] = v[i] * np.cos (theta[j])
#points=np.column_stack((vxv.ravel(), vyv.ravel(), vzv.ravel()))

vx = np.linspace(-vmax, vmax, 50)
grid_x, grid_y, grid_z= np.meshgrid(vx,vx,vx, indexing='ij')

Nx,Ny,Nz= grid_x.shape
Ntimes=fpi.shape[0]
fpicart=np.zeros((Ntimes,Nx,Ny,Nz))
for itime in range(0, 3): #Ntimes):
    fpi1=fpi[itime,:].ravel();
    print('fpi1',fpi1.shape, points.shape)
    fpicart[itime,:,:,:] = griddata(points, fpi1, (grid_x, grid_y, grid_z), method='linear')
    print(fpicart.shape)

filename = 'outto_e.h5'
file = h5py.File(filename,'w')
size = fpi.shape
file.create_dataset('fpi',size,data=fpi)
size = phi[0,:].shape
file.create_dataset('phi',size,data=phi[0,:])
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
tunix = tt.astype('int64')
size = tunix.shape
file.create_dataset('times_unix',size,data=tunix)
file.close()
                

