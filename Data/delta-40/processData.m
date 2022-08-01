%% READ DATA

% position data
filename = 'delta-40.x';
fid = fopen(filename,'r');

dim_x = fread(fid,3,'int32','b');
X = fread(fid,prod(dim_x),'single','b');
Y = fread(fid,prod(dim_x),'single','b');
Z = fread(fid,prod(dim_x),'single','b');
fclose(fid);

% flow data
filename = 'delta-40.q';
fid = fopen(filename,'r');

dim_q = fread(fid,3,'int32','b');
MACH  = fread(fid,1,'single','b');
ALPHA = fread(fid,1,'single','b');
RE    = fread(fid,1,'single','b');
TIME  = fread(fid,1,'single','b');

P = fread(fid,prod(dim_q),'single','b'); % pressure
U = fread(fid,prod(dim_q),'single','b'); % velocities
V = fread(fid,prod(dim_q),'single','b');
W = fread(fid,prod(dim_q),'single','b');
E = fread(fid,prod(dim_q),'single','b'); % energy
fclose(fid);

%% REGULAR GRID
% volume size
Lx = 2.5;
Ly = 0.8;
Lz = 0.8;
% volume resolution
Nx = 128;
Ny = 41;
Nz = 41;
% volume center
Cx = 1;
Cy = 0.38;
Cz = 0.25;

% generate grid
x1d = linspace(Cx-Lx/2,Cx+Lx/2,Nx);
y1d = linspace(Cy-Ly/2,Cy+Ly/2,Ny);
z1d = linspace(Cz-Lz/2,Cz+Lz/2,Nz);

[xx,yy,zz] = ndgrid(x1d,y1d,z1d);
xx = xx(:);
yy = yy(:);
zz = zz(:);


%% INTERPOLATE VELOCITY
uu = griddata(X,Y,Z,U,xx,yy,zz,'linear');
vv = griddata(X,Y,Z,V,xx,yy,zz,'linear');
ww = griddata(X,Y,Z,W,xx,yy,zz,'linear');