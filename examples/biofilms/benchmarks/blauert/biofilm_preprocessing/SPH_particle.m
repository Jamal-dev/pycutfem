function SPH_particle(xv, yv, bord_x, bord_y)
%
% This function generate input files for the SPH code
%
% Only for 2D problems
%
% Copyright @ Dianlei Feng

% Get the max&min coordinates of the domain [m]
x_min = min(bord_x)*1e-6;
x_max = max(bord_x)*1e-6;
y_min = min(bord_y)*1e-6;
y_max = max(bord_y)*1e-6;

% x_line = 0e-4;
v0 = 0.0684;     % Re = 91

xv = xv*1e-6;
yv = yv*1e-6;

Nx = 550%660%330;           % Number of particles along x- direction
Ny = 101%120%60;            % Number of particles along y- direction

Dx = (x_max-x_min)/Nx;
Dy = (y_max-y_min)/(Ny-1);

% Fluid particle Information
N_Fluid = Nx*Ny;
% N_Solid_temp = (Nx-1)*(Ny-1);

x_Fluid = zeros(1,(N_Fluid));
y_Fluid = x_Fluid;
m_Fluid = x_Fluid;
i_type = x_Fluid;
velx_Fluid = x_Fluid;
vely_Fluid = x_Fluid;
p_Fluid = x_Fluid;
e_Fluid = x_Fluid;


rho_Fluid = 1000;   %kg/m^3
rho_Solid = 1500;   %kg/m^3

r_Fluid = rho_Fluid * ones(size(x_Fluid));

phi = 0.47;         % Biofilm porosity ("Time-Resolved Biofilm Deformation Measurements
                    % Using Optical Coherence Tomography")
u_avg = 0.0455;    % m/s; Re = 91;                     

mass_Fluid = rho_Fluid * Dx * Dy;
mass_Solid = rho_Solid * Dx * Dy;

hsml= Dx;%1.0*sqrt(Dx^2 + Dy^2);

for i = 1 : Nx
    for j = 1 : Ny
        indx = j + (i-1)*Ny;
        x_Fluid(indx) = (i-1)*Dx;
%         y_Fluid(indx) = (j-1)*Dy + 1*Dy/2;
        y_Fluid(indx) = (j-1)*Dy + 0*Dy/2;
        
        
        [in,on] = inpolygon(x_Fluid(indx), y_Fluid(indx),xv,yv);

        if (in || on)
            m_Fluid(indx) = mass_Fluid * phi;
            i_type(indx) = 11;
        else
            m_Fluid(indx) = mass_Fluid;
            i_type(indx) = 10;
        end
        
    end
end

% Par_indx = 1 : Nx*Ny;
% Solid particle information
N_Solid = (Nx-1)*Ny;%(Ny-1);
x_Solid = zeros(1,(N_Solid));
y_Solid = x_Solid;
kk = 0;

for i = 1  : Nx-1
    for j = 1 : Ny-1
        indx = j + (i-1)*Ny;
        x_Solid(indx) = (i-1)*Dx + Dx/2;
        y_Solid(indx) = (j-1)*Dy; %+ 1*Dy/2;
        
        [in,on] = inpolygon(x_Solid(indx), y_Solid(indx),xv,yv);
        
        if (in || on)
            kk = kk + 1;
%             Par_temp = Par_indx(end) + 1;
%             Par_indx = [Par_indx, Par_temp];
            
            x_Solid(kk) = x_Solid(indx);
            y_Solid(kk) = y_Solid(indx);
        end
    end 
end


for jj = 1 : length(x_Fluid)
% if(x_Fluid(jj)<=x_line) 
if (x_Fluid(jj) > 3.5e-3)
    x_Fluid(jj) = 3.5e-3 - x_Fluid(jj);
end

if(i_type(jj)<20) 
%         velx_Fluid(jj) = v0 * (1-(2*(y_Fluid(jj) - y_max/2)/y_max)^2);
        velx_Fluid(jj) = v0;
        vely_Fluid(jj) = 0.0; 
end
end


N_Solid = kk;
x_Solid = x_Solid(1:N_Solid);
y_Solid = y_Solid(1:N_Solid);
j_type = 110*ones(1,N_Solid);
m_Solid = mass_Solid * (1 - phi) * ones(1,N_Solid);
velx_Solid = zeros(1, N_Solid);
vely_Solid = velx_Solid;
r_Solid = rho_Solid * ones(size(m_Solid))*(1-phi);
p_Solid = zeros(1, N_Solid);
e_Solid = zeros(1, N_Solid);

% Write the data information to files
ii = 1:(N_Fluid+N_Solid);

xx_min = min(x_Fluid);
%%
x_Fluid = x_Fluid - xx_min;
x_Solid = x_Solid - xx_min;

xx = [x_Fluid x_Solid];
yy = [y_Fluid y_Solid];
vx = [velx_Fluid velx_Solid];
vy = [vely_Fluid vely_Solid];
ini_vx = [ii', xx', yy', vx', vy'];

mm = [m_Fluid m_Solid];
rho = [r_Fluid r_Solid];
pp = [p_Fluid p_Solid];
ee = [e_Fluid e_Solid];
ini_state = [ii', mm', rho', pp', ee'];

itype = [i_type j_type];
hh = hsml*ones(1, (N_Fluid+N_Solid));
por = phi*ones(1, (N_Fluid+N_Solid));
por(itype==10) = 1;
ini_other =[ii', itype', hh', por'];

N_Fluid = length(x_Fluid)
N_Solid = length(x_Solid)





fileID = fopen('ini_xv.dat','w');
%fprintf(fileID,'%12s %12s %12s %12s\n','x','y', 'vx', 'vy');
fprintf(fileID,'%7d % 14.8E % 14.8E % 14.8E % 14.8E\n',ini_vx');
fclose(fileID);

fileID = fopen('ini_state.dat','w');
%fprintf(fileID,'%12s %12s %12s %12s\n','x','y', 'vx', 'vy');
fprintf(fileID,'%7d % 14.8E % 14.8E % 14.8E % 14.8E\n',ini_state');
fclose(fileID);

fileID = fopen('ini_other.dat','w');
%fprintf(fileID,'%12s %12s %12s %12s\n','x','y', 'vx', 'vy');
fprintf(fileID,'%7d % 6d % 14.8E % 14.8E\n',ini_other');
fclose(fileID);


xb_min = min(xv(yv<=5*1e-6)); % x- coordinate of the starting point of the biofilm
xb_max = max(xv);
yb_max = max(yv(xv>=xb_max - 5*1e-6));

% Coordinates of the virtual particles
%
Vcord = Virtual_particle(xx, yy, Dx, Dy, xb_min, yb_max, itype);


% xx_vir = [Vcord.xx_rep_low_F, Vcord.xx_rep_up_F, Vcord.xx_rep_low_S, Vcord.xx_rep_right_F, Vcord.xx_rep_right_S,...
%     Vcord.xx_gho_low_F, Vcord.xx_gho_up_F, Vcord.xx_gho_low_S, Vcord.xx_gho_right_F, Vcord.xx_gho_right_S];
% yy_vir = [Vcord.yy_rep_low_F, Vcord.yy_rep_up_F, Vcord.yy_rep_low_S, Vcord.yy_rep_right_F, Vcord.yy_rep_right_S,...
%     Vcord.yy_gho_low_F, Vcord.yy_gho_up_F, Vcord.yy_gho_low_S, Vcord.yy_gho_right_F, Vcord.yy_gho_right_S];
xx_vir = [Vcord.xx_rep_low_F, Vcord.xx_rep_up_F, Vcord.xx_rep_low_S,...
    Vcord.xx_gho_low_F, Vcord.xx_gho_up_F, Vcord.xx_gho_low_S];
yy_vir = [Vcord.yy_rep_low_F, Vcord.yy_rep_up_F, Vcord.yy_rep_low_S,...
    Vcord.yy_gho_low_F, Vcord.yy_gho_up_F, Vcord.yy_gho_low_S];
% Number of virtual particles
% N_rep = length(Vcord.xx_rep_low_F) + length(Vcord.xx_rep_up_F) + length(Vcord.xx_rep_low_S) + length(Vcord.xx_rep_right_F) + length(Vcord.xx_rep_right_S);
% N_gho = length(Vcord.xx_gho_low_F) + length(Vcord.xx_gho_up_F) + length(Vcord.xx_gho_low_S) + length(Vcord.xx_gho_right_F) + length(Vcord.xx_gho_right_S);
N_rep = length(Vcord.xx_rep_low_F) + length(Vcord.xx_rep_up_F) + length(Vcord.xx_rep_low_S) ;
N_gho = length(Vcord.xx_gho_low_F) + length(Vcord.xx_gho_up_F) + length(Vcord.xx_gho_low_S) ;
% Velocity
vx_vir = zeros(1, N_rep+N_gho);
vy_vir = zeros(1, N_rep+N_gho);


% 
% itype_vir  = [20 * ones(size(Vcord.xx_rep_low_F)), 20 * ones(size(Vcord.xx_rep_up_F)), ...
%     200 * ones(size(Vcord.xx_rep_low_S)), ...
%     30 * ones(size(Vcord.xx_gho_low_F)), 30 * ones(size(Vcord.xx_gho_up_F)), ...
%     300 * ones(size(Vcord.xx_gho_low_S)), ...
%     ];

itype_vir  = [30 * ones(size(Vcord.xx_rep_low_F)), 30 * ones(size(Vcord.xx_rep_up_F)), ...
    300 * ones(size(Vcord.xx_rep_low_S)), ...
    30 * ones(size(Vcord.xx_gho_low_F)), 30 * ones(size(Vcord.xx_gho_up_F)), ...
    300 * ones(size(Vcord.xx_gho_low_S)), ...
    ];

% itype_vir((xx_vir<xb_min)&(yy_vir<yb_max))=30;

x_refl = 3.5e-3;

for iv = 1 : length(xx_vir)
    if ( xx_vir(iv) > x_refl )&&( itype_vir(iv) <100 )
        dist =  xx_vir(iv)-x_refl;
        
        if ((yy_vir(kk)> y_max*0.8)&&(itype_vir(kk)==30)) 
           xx_vir = [xx_vir, -1*dist, -1*dist-Dx/2];
           yy_vir = [yy_vir, yy_vir(iv), yy_vir(iv)];
        
           vx_vir = [vx_vir, vx_vir(iv), vx_vir(iv)];
           vy_vir = [vy_vir, vy_vir(iv), vy_vir(iv)];
           
           itype_vir = [itype_vir, itype_vir(iv), itype_vir(iv)];
        else  
            xx_vir = [xx_vir, -1*dist];
            yy_vir = [yy_vir, yy_vir(iv)];
        
            vx_vir = [vx_vir, vx_vir(iv)];
            vy_vir = [vy_vir, vy_vir(iv)];
        
            itype_vir = [itype_vir, itype_vir(iv)];
        end
        
    end
end
        

p_vir = zeros(size(itype_vir));
e_vir = zeros(size(itype_vir));
m_vir = zeros(size(itype_vir));
rho_vir = zeros(size(itype_vir));
por_vir = zeros(size(itype_vir));

% for kk = 1 : length(itype_vir)
%     
% %     if ((xx_vir(kk)<= xb_min) && (itype_vir(kk) == 20 || itype_vir(kk) == 30))
% %     if ( ((xx_vir(kk)<= xb_min) && (itype_vir(kk) == 30)) || ((yy_vir(kk)> y_max*0.8) && (itype_vir(kk) == 30)) )
% %         if ( ((xx_vir(kk)<= 2.0e-3) && (itype_vir(kk) == 30)) || ((yy_vir(kk)> y_max*0.8) && (itype_vir(kk) == 30)) )
%             if ( ((xx_vir(kk)<= 2.9e-3 || xx_vir(kk)>= 4.5e-3) && (itype_vir(kk) == 30)) || ((yy_vir(kk)> y_max*0.8) && (itype_vir(kk) == 30)) )
%         m_vir(kk) = mass_Fluid;
%         
%         rho_vir (kk) = rho_Fluid;
%         por_vir (kk) = 1.0;
%         hsml_vir(kk) = Dx;%sqrt((Dx/2)^2+(Dy)^2);
%     else
%         por_vir (kk) = phi;
%         hsml_vir(kk) = hsml;
%         if (itype_vir(kk) <= 100)
%             m_vir(kk) = mass_Fluid * phi;
%             rho_vir (kk) = rho_Fluid;
%         elseif (itype_vir(kk) > 100)
%             m_vir(kk) = mass_Solid * (1-phi);
%             rho_vir (kk) = rho_Solid * (1-phi);
%         end
%     end
%     
% end
% % Smoothing length
% hsml_vir = hsml * ones(1, N_rep + N_gho);

% xx_vir_ = [Vcord.xx_rep_low_F, Vcord.xx_rep_up_F, Vcord.xx_rep_low_S, Vcord.xx_rep_right_F, Vcord.xx_rep_right_S,...
%     Vcord.xx_gho_low_F, Vcord.xx_gho_up_F, Vcord.xx_gho_low_S];
% yy_vir_ = [Vcord.yy_rep_low_F, Vcord.yy_rep_up_F, Vcord.yy_rep_low_S, Vcord.yy_rep_right_F, Vcord.yy_rep_right_S,...
%     Vcord.yy_gho_low_F, Vcord.yy_gho_up_F, Vcord.yy_gho_low_S];
xx_vir = xx_vir - xx_min; 

for kk = 1 : length(itype_vir)
    
%     if ((xx_vir(kk)<= xb_min) && (itype_vir(kk) == 20 || itype_vir(kk) == 30))
%     if ( ((xx_vir(kk)<= xb_min) && (itype_vir(kk) == 30)) || ((yy_vir(kk)> y_max*0.8) && (itype_vir(kk) == 30)) )
%         if ( ((xx_vir(kk)<= 2.0e-3) && (itype_vir(kk) == 30)) || ((yy_vir(kk)> y_max*0.8) && (itype_vir(kk) == 30)) )
            if ( ((xx_vir(kk)<= 3.07e-3 || xx_vir(kk)>= 4.49e-3) && (itype_vir(kk) == 30)) || ((yy_vir(kk)> y_max*0.8) && (itype_vir(kk) == 30)) )
        m_vir(kk) = mass_Fluid;
        
        rho_vir (kk) = rho_Fluid;
        por_vir (kk) = 1.0;
        hsml_vir(kk) = Dx;%sqrt((Dx/2)^2+(Dy)^2);
    else
        por_vir (kk) = phi;
        hsml_vir(kk) = Dx;
        if (itype_vir(kk) <= 100)
            m_vir(kk) = mass_Fluid * phi;
            rho_vir (kk) = rho_Fluid;
        elseif (itype_vir(kk) > 100)
            m_vir(kk) = mass_Solid * (1-phi);
            rho_vir (kk) = rho_Solid * (1-phi);
        end
    end
    
end

N_vir = length(xx_vir)

figure (2)
plot(x_Fluid, y_Fluid, 'go','MarkerSize',2, 'MarkerFaceColor', 'g');
hold on
plot(x_Solid, y_Solid, 'ro','MarkerSize',2, 'MarkerFaceColor', 'r');
hold on
plot(xx_vir, yy_vir, 'bs','MarkerSize',2, 'MarkerFaceColor', 'b');
hold on
plot(xx_vir(itype_vir==300), yy_vir(itype_vir==300), 'k','MarkerSize',2);

figure (3)

%  scatter(xx_vir',yy_vir',[], itype_vir','filled');
%  hold on
%  scatter(xx, yy, [], itype,'filled');
 scatter(xx, yy, [], mm,'filled');
% Write the data into files

kk = 1 : length(m_vir);
% por_vir = phi*ones(1, length(m_vir));
vx_vp = [kk', xx_vir', yy_vir', vx_vir', vy_vir'];

state_vp = [kk', m_vir', rho_vir', p_vir', e_vir'];

other_vp =[kk', itype_vir', hsml_vir', por_vir'];

fileID = fopen('xv_vp.dat','w');
%vx_vp = [kk', xx_vir', yy_vir', vx_vir', vy_vir'];
fprintf(fileID,'%7d % 14.8E % 14.8E % 14.8E % 14.8E\n',vx_vp');
fclose(fileID);

fileID = fopen('state_vp.dat','w');
%state_vp = [kk', m_vir', rho_vir', p_vir', e_vir'];
fprintf(fileID,'%7d % 14.8E % 14.8E % 14.8E % 14.8E\n',state_vp');
fclose(fileID);

figure (4)
 scatter(xx_vir, yy_vir, [], m_vir,'filled');
fileID = fopen('other_vp.dat','w');
%other_vp =[kk', itype_vir', hsml_vir', por_vir'];
fprintf(fileID,'%7d % 6d % 14.8E % 14.8E\n',other_vp');
fclose(fileID);

