clear all
close all
addpath(genpath('~/Data/codes/matanak/'))
load gist_ncar.mat

fn='outto.h5'
hinfo=hdf5info(fn);
fpi=hdf5read(fn,'/fpi/');
phi=hdf5read(fn,'/phi/')-pi;
theta=hdf5read(fn,'/theta/');
energy=hdf5read(fn,'/energy/');
grid_x = hdf5read(fn,'/grid_x/');
grid_y = hdf5read(fn,'/grid_y/');
grid_z = hdf5read(fn,'/grid_z/');
fcart = hdf5read(fn,'/fpcart/');

xgrid= squeeze(grid_x(1,1,:));
ygrid= squeeze(grid_y(1,:,1));
zgrid= squeeze(grid_z(:,1,1));

for itime=1:size(fcart,4)

    itime
Np=1e5;
[xp, yp, zp] = generate_particles(grid_x , grid_y , grid_z, fcart(:,:,:,itime),Np);

figure()
 scatter3(xp(1:100:Np),yp(1:100:Np),zp(1:100:Np))
 daspect([1 1 1])

X=[xp;yp;zp]';
options = statset('MaxIter',1000);

NGMM=20;
for i=1:NGMM
GMModel = fitgmdist(X,i,'RegularizationValue',0.1,'Options',options);
%GMModel.ComponentProportion
%GMModel.mu
AIC(itime,i)=GMModel.AIC;
BIC(itime,i)=GMModel.BIC;
  figure(1)
  hold on
  xc=GMModel.mu(:,1)
  yc=GMModel.mu(:,2)
  zc=GMModel.mu(:,3)
  plot3(xc,yc,zc,'*')
end
figure(2)
subplot(2,1,1)
plot(1:NGMM,AIC(itime,:))
title('AIC')
axis tight
xlabel('number of clusters','fontsize',15)
set(gca,'fontsize',15)
subplot(2,1,2)
plot(1:NGMM,BIC(itime,:))
axis tight
title('BIC')
xlabel('number of clusters','fontsize',15)
set(gca,'fontsize',15)
print('-dpng','aicbic.png')


AIC(itime,:)=AIC(itime,:)/AIC(itime,1);
BIC(itime,:)=BIC(itime,:)/BIC(itime,1);
figure
subplot(2,2,1)
imagesc(AIC(:,2:end)')
subplot(2,2,2)
imagesc(log10(AIC(:,2:end)'))
subplot(2,2,3)
imagesc(BIC(:,2:end)')
subplot(2,2,4)
imagesc(log10(BIC(:,2:end)'))
colormap(gist_ncar)
pause
end
figure
imagesc(AIC(:,2:end)')
set(gca,'fontsize',15)
colormap(gist_ncar)
colorbar
xlabel('Time','fontsize',15)
ylabel('Number of Gaussians','fontsize',15)
print('-dpng','AIC.png')

