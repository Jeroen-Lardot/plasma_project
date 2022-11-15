import pytplot
import pyspedas
import h5py
from scipy.interpolate import griddata
import numpy as np

def write_vtk(ds_arr,x,y,z,itime):
    [nx, ny, nz] = np.shape(ds_arr)
    a=nx*ny*nz
    f = open('../bic/'+'VTK_data'+str(itime)+'.vtk','w') # change your vtk file name
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
    #f.close()
    #f = open('VTK_data3.vtk','ab') # change your vtk file name
    #f.write(ds_arr.tobytes())
    #out=ds_arr.flatten()
    #out.tofile(f)
    #ds_arr.ndarray.tofile(f, sep='', format='%s')
    for i in range(0,nx):
       for j in range(0,ny):
           for k in range(0,nz):
              content = str(ds_arr[i,j,k])
              f.write(content+'\n')
    f.close()
    return


time = '2016-01-21/01:06:50.799'
vmax = 600
mi = 1.67e-27
e = 0.5*mi*(vmax*1e3)**2
e = e/(1.6e-19)
 
#mms_part_slice2d(probe = '1', interpolation='2d', time=time, instrument='fpi', data_rate = 'brst', species='i', rotation='bv', erange=[0, e])


mms_dist=pyspedas.mms.fpi(probe = '3', trange=[time, time],datatype='dis-dist', data_rate = 'brst')
fpi=pytplot.data_quants['mms3_dis_dist_brst'].values
print('fpi',fpi.shape)
phid=pytplot.data_quants['mms3_dis_phi_brst']
print('phi',phid)
#phi=phid['data']/360*2*np.pi
phi=phid.values/360*2*np.pi
energy=pytplot.data_quants['mms3_dis_energy_brst'].values
theta=pytplot.data_quants['mms3_dis_theta_brst']['data']/360*2*np.pi
tt=pytplot.data_quants['mms3_dis_dist_brst'].time
print(tt)
tt=tt.values
print(tt)
v =np.sqrt(2 / mi * energy[0,:]*(1.6e-19)) *1e-3;
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
for itime in range(0, Ntimes):
    fpi1=fpi[itime,:].ravel();
    print('fpi1',fpi1.shape, points.shape)
    fpicart[itime,:,:,:] = griddata(points, fpi1, (grid_x, grid_y, grid_z), method='linear')
    print(fpicart.shape)
    write_vtk(fpicart[itime,:,:,:],grid_x, grid_y, grid_z,itime)

filename = 'outto_tst.h5'
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
size = tt.shape
file.create_dataset('times',size,data=tt)
file.close()
                

