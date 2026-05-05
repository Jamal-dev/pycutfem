 function data_processing
%
% Read data from "biofilm1.txt"
%
% Generate the data files for the SPH simulation

fileID = fopen('biofilm.txt','r');

formatSpec = '%f %f';

sizeA = [2 Inf];
A = fscanf(fileID,formatSpec, sizeA);

%% Read x- and y- coordinates
vx_file = A(1, :);
vy_file = A(2, :);

% x_min = 0;
x_max = max(vx_file);
% y_min = 0;
y_max = max(vy_file);

% Normolization the coordinates
vx = vx_file;
vy = y_max - vy_file;

plot (vx, vy);

%% Transfer to physical coordinates
% % % L = 1886.75 - 19.25;
% % % dL = 1852.25 - 1616.75;
% % % % L = 1891 - 19;
% % % % dL = 1852 - 1618;
% % % ruler = 250; %um
% % % L_mu = L/dL*ruler
L_mu = 2000; %Length of the domain (um)
vx_mu = vx/x_max * L_mu;
vy_mu = vy/x_max * L_mu;

x_max_mu = L_mu;
% y_max_mu = L_mu*3/8;
y_max_mu = L_mu*4/8;
x_min_mu = 0;
y_min_mu = 0;

shift = 500; % um
vx_mu = vx_mu + shift;
x_max_mu = x_max_mu + 7*shift;

bord_x = [x_min_mu, x_min_mu, x_max_mu, x_max_mu, x_min_mu];
bord_y = [y_min_mu, y_max_mu, y_max_mu, y_min_mu, y_min_mu];

vx_mu = [vx_mu vx_mu(1)];
vy_mu = [vy_mu vy_mu(1)];

figure (1)
plot(vx_mu, vy_mu);
hold on
plot(bord_x, bord_y, 'g');
axis equal 

SPH_particle(vx_mu, vy_mu, bord_x, bord_y);
