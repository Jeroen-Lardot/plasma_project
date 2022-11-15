function [xp, yp,  zp] = generate_particles(vvx,vvy,vvz,fcutoff, Np)


[nx,ny,nz]=size(fcutoff);
%vmax=560;                             %for small box
%vmax=560*3.75;                       %for big box

dvx=vvx(1,1,2)-vvx(1,1,1);
dvy=vvy(1,2,1)-vvy(1,1,1);
dvz=vvz(2,1,1)-vvz(1,1,1);

%dv=2*vmax/(nx-1);
%v1D=-vmax:dv:vmax;
%[vvx,vvy,vvz]=meshgrid(v1D,v1D,v1D);
f1D=fcutoff(:);

fmin=min(f1D(f1D>0));
fmax=max(f1D);
%f1D(f1D<fmax/100)=0;

%Distribution cannot equal zero for method to work
%so set zeros (if the exist) to small fraction of 
%smallest nonzero value.
f1D(f1D==0)=fmin;      
vx1D=vvx(:)'; vy1D=vvy(:)'; vz1D=vvz(:)';


Ng=size(f1D,1);
ranarr=rand(4,Np);
fcum=cumsum(f1D);
fcum=Ng*fcum/fcum(Ng);
Pg=interp1(fcum',1:Ng,Ng*ranarr(1,:));
Pg=1+floor(Pg);
xp=vx1D(Pg)+dvx*ranarr(2,:)-dvx/2;
yp=vy1D(Pg)+dvy*ranarr(3,:)-dvy/2;
zp=vz1D(Pg)+dvz*ranarr(4,:)-dvz/2;

end
