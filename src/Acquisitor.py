
import pyspedas as psp
import pytplot as ptp
import numpy as np
import scipy.interpolate as ipt
import aidapy as ap
import scipy.interpolate as ipt
import h5py
import matplotlib.pyplot as plt
import itertools
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm
from cycler import cycler

class Acquisitor():
    _mi = 1.67e-27
    _xr_mms = None
    _e = 0
    def __init__(self, vmax: int = 600, probes: int = 3, grid_dim: int = 50, n_components_range: int = 17, n_part: int = 40000, information_criterion: str = 'bic', write_vtk: bool = False, write_h5: bool = False) -> None:
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

    def get_data(self, t_start: str, t_end: str, mms_analysis: bool) -> None:
        settings = {'prod': ['i_dist'], 'probes': self.probes, 'coords': 'gse', 'mode': 'high_res', 'frame':'gse'}
        self._xr_mms = ap.load_data(mission='mms', start_time=t_start, end_time=t_end, **settings)
        self.save_data(mms_analysis)

    def save_data(self, mms_analysis) -> None:
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
        Ntimes = int(Ntimes)
        fpicart=np.zeros((Ntimes,Nx,Ny,Nz))
        for itime in range(0, Ntimes):
            print(f"{itime/Ntimes*100}%")
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
        if(mms_analysis):
            print("Starting MMS Analysis:")
            ### various vdf plots ###


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
                Np=self.n_part

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
                print(f"GMM {i}")
                print(gmmdata)
                for n_components in range (1, self.n_components_range):

                    gmm = GaussianMixture(n_components,covariance_type='full' ,reg_covar=0.1, init_params='kmeans', random_state=np.random.RandomState(seed=1234)).fit(gmmdata)
                    if (self.information_criterion=='aic'): info_crit.append(gmm.aic(gmmdata)) 
                    elif (self.information_criterion=='bic'): info_crit.append(gmm.bic(gmmdata)) 
                    
                    if info_crit[-1] < lowest_info_crit:
                            lowest_info_crit = info_crit[-1]
                            best_gmm = gmm
                
                info_crit = np.array(info_crit)
                color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
                clf = best_gmm
                bars = []

                #bic vs nb_components plot
                fig, ax = plt.subplots(1,1)
                plt.title('time '+str(i))
                plt.scatter(range(1, self.n_components_range), info_crit)
                plt.xlabel('number of components')       
                plt.ylabel(self.information_criterion)
                fig.savefig(f'plots/bic_vs_nbcomp_time{i}.png')
                plt.close()

                fcm_labels = best_gmm.predict(gmmdata)
                nclusters_plot.append(best_gmm.n_components)
                info_plot.append(info_crit)
                print('probe:',self.probes, 'vdf:',i,'n_particles:',self.n_part,'info:',self.information_criterion,'gmm:',best_gmm.n_components, best_gmm.covariance_type)

                ini = clf.means_
                colors = ["navy"]*len(ini)
                fig = plt.figure()
                mycycler = (cycler(color=['blue', 'orange', 'green', 'red','purple', 'brown', 'pink', 'gray', 'olive', 'cyan']))
                plt.suptitle('time '+str(i))

                plt.rc('axes', prop_cycle=mycycler)
                ax = fig.add_subplot(121,projection='3d')
                for j, color in enumerate(colors):
                    data = gmmdata[clf.predict(gmmdata) == j]
                    ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='.', alpha=0.1)
                    #print(data)
                ax.scatter(ini[:, 0], ini[:, 1], ini[:,2], s=40,color='orange', lw=1, edgecolors="black")

                ax.grid()
                ax.set_xlim(-650,650)
                ax.set_ylim(-650,650)
                ax.set_zlim(-650,650)
                ax.set_xlabel('Vx (km/s)')
                ax.set_ylabel('Vy (km/s)')
                ax.set_zlabel('Vz (km/s)')

                #right panel
                ax2 = fig.add_subplot(122,projection='3d')
                ax2.scatter(ini[:, 0], ini[:, 1], ini[:,2], s=40,color='orange', lw=1, edgecolors="black")

                covs = clf.covariances_
                #print(covs)
                for k in range(best_gmm.n_components):
                    u = np.linspace(0, 2 * np.pi, 100)
                    v = np.linspace(0, np.pi, 100)

                    #needs to add 0.01 for the "1sigma" (no more) blobs to be within same v range as data 
                    x = 0.01*np.outer(np.cos(u), np.sin(v))
                    y = 0.01*np.outer(np.sin(u), np.sin(v))
                    z = 0.01*np.outer(np.ones_like(u), np.cos(v))

                    #I admit, the 10000* is wierd, but otherwise, he complains about the shape and now it seems to work
                    #(unless maybe that the 1sigma regions are wierdly large)
                    bias = np.array([10000*[ini[k][0]], 10000*[ini[k][1]], 10000*[ini[k][2]]])
                    ellipsoid = (covs[k] @ np.stack((x, y, z), 0).reshape(3, -1) + bias).reshape(3, *x.shape)
                    ax2.plot_surface(*ellipsoid,  rstride=4, cstride=4, linewidth=0, alpha=0.2)

                ax2.grid()
                ax2.set_xlim(-650,650)
                ax2.set_ylim(-650,650)
                ax2.set_zlim(-650,650)
                ax2.set_xlabel('Vx (km/s)')
                ax2.set_ylabel('Vy (km/s)')
                ax2.set_zlabel('Vz (km/s)')
                plt.tight_layout()
                fig.savefig(f'plots/3d_plots_time{i}.png',dpi=150)
                plt.close()
                #exit()
                ### plot integrating over 1 axis ###
            #     fcut=fpicart[i,:,:,:]
            #     ftot1=np.sum(fcut, axis=0)
            #     print(ftot1)
            #     ftot2=np.sum(fcut, axis=1)
            #     ftot3=np.sum(fcut, axis=2)
            #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            #     fig.suptitle('fpi '+str(i)+' plan: xy,zx,yz')
            #     ax1.imshow(ftot1, cmap='jet')
            #     ax1.scatter(ini[:, 0], ini[:, 1], s=75, marker="D", c="orange", lw=1.5, edgecolors="black")
            #     ax2.imshow(ftot2, cmap='jet')
            #     ax3.imshow(ftot3, cmap='jet')
            #     #plt.colorbar(scat)
            #     plt.show()



            plt.clf()        
            plt.plot(nclusters_plot,'bo--')
            plt.title('probe '+str(self.probes)+' n_particles '+str(self.n_part)+' info '+self.information_criterion)
            plt.ylabel('gmm ecomponents')
            plt.xlabel('time')
            plt.savefig(f'plots/nbcomp_vs_time.png',dpi=150)
            plt.show()
            plt.close()

            # ### plot aic/bic slope ###
            # cf=plt.imshow(np.transpose(info_plot)/np.amax(info_plot), origin = 'upper', extent=[0,itime,self.n_components_range-1,1], cmap='jet', aspect='auto')

            # plt.xlabel('time')
            # plt.ylabel('n of clusters')
            # # plt.colorbar(cf,format='%.10f')
            # plt.colorbar()
            # plt.show()

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
