
import pyspedas as psp
import pytplot as ptp
import numpy as np
import scipy.interpolate as ipt
import aidapy as ap
import scipy.interpolate as ipt
import h5py

class Acquisitor():
    _mi = 1.67e-27
    _xr_mms = None
    _e = 0
    def __init__(self, vmax: int = 600, probes: int = 3, grid_dim: int = 50, n_components_range: int = 15, n_part: int = 40000, information_criterion: str = 'bic', write_vtk: bool = False, write_h5: bool = False) -> None:
        self._vmax = vmax
        self._probes = probes
        self._grid_dim = grid_dim
        self._n_components_range = n_components_range
        self._n_part = n_part
        self._information_criterion = information_criterion
        self._write_vtk = write_vtk
        self._write_h5 = write_h5
        self._e = (0.5*self._mi*(self._vmax*1e3)**2) / (1.6e-19)

    @property
    def vmax(self) -> int:
        return self._vmax
       
    @vmax.setter
    def vmax(self, vmax) -> None:
        if(vmax < 0):
           raise ValueError("vmax should be a positive integer.")
        self._e = (0.5*self._mi*(self._vmax*1e3)**2) / (1.6e-19)
        self._vmax = vmax

    @property
    def mms_dist(self) -> object:
        if self._mms_dist == None:
            raise ValueError("mms distribution has not been initialized yet.")
        return self._mms_dist
       
    @mms_dist.setter
    def mms_dist(self, mms_dist) -> None:
        self._mms_dist = mms_dist

    @property
    def probes(self) -> int:
        return self._probes
       
    @probes.setter
    def probes(self, probes) -> None:
        if(probes < 1):
            raise ValueError("You should have at least one probe.")
        self._probes = probes

    @property
    def grid_dim(self) -> int:
        return self._grid_dim

    @grid_dim.setter
    def grid_dim(self, grid_dim) -> None:
        if (4 > grid_dim and grid_dim > 0):
            raise ValueError("Dimension should be between 1 and 3.")
        self._grid_dim = grid_dim

    @property
    def n_components_range(self) -> int:
        return self._n_components_range
       
    @n_components_range.setter
    def n_components_range(self, n_components_range) -> None:
        self._n_components_range = n_components_range

    @property
    def n_part(self) -> int:
        return self._n_part
       
    @n_part.setter
    def n_part(self, n_part) -> None:
        self._n_part = n_part

    @property
    def information_criterion(self) -> int:
        return self._information_criterion
       
    @information_criterion.setter
    def information_criterion(self, information_criterion) -> None:
        if(information_criterion != 'bic' and information_criterion != 'aic'):
            raise ValueError("Received an unknown information criterion.")
        self._information_criterion = information_criterion

    @property
    def write_vtk(self) -> int:
        return self._write_vtk
       
    @write_vtk.setter
    def write_vtk(self, write_vtk) -> None:
        if(not isinstance(write_vtk, bool)):
            raise ValueError("write_vtk has to be a boolean value.")
        self._write_vtk = write_vtk

    @property
    def write_h5(self) -> int:
        return self._write_h5
       
    @write_h5.setter
    def write_h5(self, write_h5) -> None:
        if(not isinstance(write_h5, bool)):
            raise ValueError("write_h5 has to be a boolean value.")
        self._write_h5 = write_h5

    def get_data(self, t_start: str, t_end: str) -> None:
        settings = {'prod': ['i_dist'], 'probes': self.probes, 'coords': 'gse', 'mode': 'high_res', 'frame':'gse'}
        self._xr_mms = ap.load_data(mission='mms', start_time=t_start, end_time=t_end, **settings)
        self.save_data()

    def save_data(self) -> None:
        phistr='mms{}_dis_phi_brst'.format(self.probes)
        thetastr='mms{}_dis_theta_brst'.format(self.probes)
        energystr='mms{}_dis_energy_brst'.format(self.probes)
        fpistr='i_dist{}'.format(self.probes)

        fpi=np.array(self._xr_mms[fpistr])
        phi=np.array(self._xr_mms[phistr]/360*2*np.pi)
        theta=np.array(self._xr_mms[thetastr]/360*2*np.pi)
        energy=np.array(self._xr_mms[energystr])

        tt=np.array(self._xr_mms['time1']).astype('<i8')

        v =np.sqrt(2/self._mi*energy*(1.6e-19)) *1e-3
        fiv, thv, vv = np.meshgrid(phi, theta, v,  indexing='ij')
        vxv = vv * np.cos (fiv) * np.sin (thv)
        vyv = vv * np.sin (fiv) * np.sin (thv)
        vzv = vv * np.cos (thv)

        points=np.column_stack((vxv.ravel(), vyv.ravel(), vzv.ravel()))
        vx = np.linspace(-self.vmax, self.vmax, self.grid_dim)
        grid_x, grid_y, grid_z= np.meshgrid(vx,vx,vx, indexing='ij')
        Nx,Ny,Nz= grid_x.shape
        Ntimes=fpi.shape[0]
        fpicart=np.zeros((Ntimes,Nx,Ny,Nz))

        for itime in range(0, Ntimes):
            fpi1=fpi[itime,:].ravel()
            fpicart[itime,:,:,:] = ipt.griddata(points, fpi1, (grid_x, grid_y, grid_z), method='linear')
        
            ### writing vtk file with distributions for paraview ###
            if(self.write_vtk == True):
                self.write_to_vtk(fpicart[itime,:,:,:],grid_x, grid_y, grid_z,itime)

        if (self.write_h5 == True):
            file = h5py.File('outto_tst.h5','w')
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

    def write_to_vtk(self, ds_arr,x,y,z,itime):
        [nx, ny, nz] = np.shape(ds_arr)
        f = open(f"../{self.information_criterion}/VTK_data{itime}.vtk", "w")
        self.write_header(f, [x, y, z], [nx, ny, nz])

        for i in range(0,nx):
            for j in range(0,ny):
                for k in range(0,nz):
                    content = str(ds_arr[i,j,k])
                    f.write(content+'\n')
        f.close()
        return

    def write_header(self, f, location, nlocation) -> None:
        [x, y, z], [nx, ny, nz] = location, nlocation
        f.write('# vtk DataFile Version 2.0\n')
        f.write('ASCII\n')
        f.write('DATASET STRUCTURED_POINTS\n')
        f.write(f'DIMENSIONS {nx} {ny} {nz}\n') # change your dimension
        f.write(f'SPACING {x[1,0,0]-x[0,0,0]} {y[0,1,0]-y[0,0,0]} {z[0,0,1]-z[0,0,0]}\n')
        f.write(f'ORIGIN {x[0,0,0]} {y[0,0,0]} {z[0,0,0]}\n')
        f.write(f'POINT_DATA {nx*ny*nz}\n') # change the number of point data
        f.write('SCALARS VDF float\n')
        f.write('LOOKUP_TABLE default\n')