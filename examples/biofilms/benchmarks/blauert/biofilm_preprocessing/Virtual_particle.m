function Vcord = Virtual_particle(xx, yy, Dx, Dy, xb_min, yb_max, i_type)
%
% This function generates the boundary particles
% 
% Following "On the treatment of solid boundary in smoothed particle
% hydrodynamics" by Moubin Liu et.al.

N_layer = 4;    % Number of lyers of the virtual particles

% Generate the repulisive particles
xx_max = max(xx);
xx_min = min(xx);
yy_max = max(yy);
yy_min = min(yy);

i_rep_low_F = 0;
i_rep_up_F = 0;
xx_rep_low_F = [];
yy_rep_low_F = [];
xx_rep_up_F = [];
yy_rep_up_F = [];

i_rep_low_S = 0;
xx_rep_low_S = [];
yy_rep_low_S = [];

i_rep_right_S = 0;
xx_rep_right_S = [];
yy_rep_right_S = []; 

i_rep_right_F = 0;
xx_rep_right_F = [];
yy_rep_right_F = []; 

% Generate the repulisive particles (Fluid & Solid)
yy_min = yy_min - 0*Dy/1;
for i = 1 : length(xx)
    % Generate the lower repulisive particles (Fluid)
%     if (yy(i)>= yy_min+1*Dy/2-Dy/4)&&(yy(i)<=yy_min+1*Dy/2+Dy/4)&&(i_type(i)<20)
     if (yy(i)>= yy_min+0*Dy/2-Dy/4)&&(yy(i)<=yy_min+0*Dy/2+Dy/4)&&(i_type(i)<20)
       i_rep_low_F = i_rep_low_F + 1;
       x_rep = xx(i);
       y_rep = yy_min - 2*Dy/2.;
       xx_rep_low_F = [xx_rep_low_F, x_rep];
       yy_rep_low_F = [yy_rep_low_F, y_rep];
           
    end
%         % Velocity
%     vx_rep_low_F = zeros(size(xx_rep_low_F));
%     vy_rep_low_F = zeros(size(yy_rep_low_F));
%     
%         % State variables (mass, density, pressure, internal energy)
%     mass_rep_low_F = zeros(size(xx_rep_low_F));
%     mass_rep_low_F(xx_rep_low_F<xb_min) = 
    
    % Generate the uper repulisive particles (Fluid)
    if (yy(i)>= yy_max-Dy/4)&&(yy(i)<=yy_max+Dy/4)
       i_rep_up_F = i_rep_up_F + 1;
       x_rep = xx(i);
%        x_rep_2 = x_rep + Dx/2;
       y_rep = yy_max + Dy/1;
%        y_rep = yy_max + Dy/1 - Dy/2.;
%        y_rep_2 = y_rep;
%        xx_rep_up_F = [xx_rep_up_F, x_rep, x_rep_2];
%        yy_rep_up_F = [yy_rep_up_F, y_rep, y_rep_2];
       
       xx_rep_up_F = [xx_rep_up_F, x_rep];
       yy_rep_up_F = [yy_rep_up_F, y_rep];
    end
    
    % Generate the lower repulisive particles (Solid)
    if (yy(i)>= yy_min-Dy/4)&&(yy(i)<=yy_min+Dy/4)&&(i_type(i)<20)
%         if (yy(i)>= yy_min-Dy/4+1*Dy/2)&&(yy(i)<=yy_min+Dy/4+1*Dy/2)&&(i_type(i)<20)
%             if (yy(i)>= yy_min-Dy/4+0*Dy/2)&&(yy(i)<=yy_min+Dy/4+0*Dy/2)&&(i_type(i)<20)
       i_rep_low_S = i_rep_low_S + 1;
       x_rep = xx(i)+ Dx/2.;
%         x_rep = xx(i);
       y_rep = yy_min - 2*Dy/2.;
       xx_rep_low_S = [xx_rep_low_S, x_rep];
       yy_rep_low_S = [yy_rep_low_S, y_rep];
    end
    
%     %  Generate the right repulisive particles (Solid)
%     if (xx(i)>= xx_max - Dx/2 - Dx/4) && (xx(i)<= xx_max - Dx/2 + Dx/4) && (yy(i)<=yb_max)
%        i_rep_right_S = i_rep_right_S + 1;
%        x_rep = xx(i) + Dx;
%        y_rep = yy(i);
%        xx_rep_right_S = [xx_rep_right_S, x_rep];
%        yy_rep_right_S = [yy_rep_right_S, y_rep];
%     end
    
end
xx_temp_rep = [xx xx_rep_low_F xx_rep_low_S];
yy_temp_rep = [yy yy_rep_low_F yy_rep_low_S];


for i =  1 : length(xx_temp_rep)
%     %  Generate the right repulisive particles (Fluid)
%     if (xx_temp_rep(i)>= xx_max - Dx/4) && (xx_temp_rep(i)<= xx_max + Dx/4) && (yy_temp_rep(i)<= max(yy_rep_right_S))
% %     if (xx(i)>= xx_max - Dx/4) && (xx(i)<= xx_max + Dx/4) && (yy(i)<= max(yy_rep_right_S))
%        i_rep_right_F = i_rep_right_F + 1;
%        x_rep = xx_temp_rep(i) + Dx/2;
%        y_rep = yy_temp_rep(i);
% %        x_rep = xx(i) + Dx/2;
% %        y_rep = yy(i);
%        xx_rep_right_F = [xx_rep_right_F, x_rep];
%        yy_rep_right_F = [yy_rep_right_F, y_rep];
%     end
%     
%     i_rep_right = i_rep_right_S + i_rep_right_F; 
end
% i_rep_low_F
% i_rep_up_F
% i_rep_low_S

i_ghost_low_F = 0;
i_ghost_low_S = 0;
i_gho_right_S = 0;
i_gho_right_F = 0;
i_ghost_up_F = 0;
xx_gho_low_F = [];
yy_gho_low_F = [];
xx_gho_up_F = [];
yy_gho_up_F = [];
xx_gho_low_S = [];
yy_gho_low_S = [];
xx_gho_right_S = [];
yy_gho_right_S = [];
xx_gho_right_F = [];
yy_gho_right_F = [];

% Generate the ghost particles at the lower boundary
for j = 1 : N_layer
    for i = 1 : length(xx)
        % Generate the lower ghost particles (Fluid)
%         if (yy(i)>= yy_min+1*Dy/2-Dy/4)&&(yy(i)<=yy_min+1*Dy/2+Dy/4)&&(i_type(i)<20)
            if (yy(i)>= yy_min-0*Dy/2-Dy/4)&&(yy(i)<=yy_min-0*Dy/2+Dy/4)&&(i_type(i)<20)
            i_ghost_low_F = i_ghost_low_F + 1;
            x_gho = xx(i);
%             y_gho = yy_min - (j) * Dy/2.;
%             if (j==1) 
                y_gho = yy_min - (j) * Dy/1. -2*Dy/2.;
%             else
%                 y_gho = yy_min - (j) * Dy/1.;
%             end
            xx_gho_low_F = [xx_gho_low_F, x_gho];
            yy_gho_low_F = [yy_gho_low_F, y_gho];
        end
        
        % Generate the uper ghost particles (Fluid)
        if (yy(i)>= yy_max-Dy/4)&&(yy(i)<=yy_max+Dy/4)
            i_ghost_up_F = i_ghost_up_F + 1;
            x_gho = xx(i);
            x_gho_2 = x_gho + Dx/2;
%             y_gho = yy_max + (j+1) * Dy/2.;
            y_gho = yy_max + (j+1) * Dy/1.;
%             y_gho = yy_max + (j+1) * Dy/1. - Dy/2.;
            y_gho_2 = y_gho;
%             xx_gho_up_F = [xx_gho_up_F, x_gho, x_gho_2];
%             yy_gho_up_F = [yy_gho_up_F, y_gho, y_gho_2];
            xx_gho_up_F = [xx_gho_up_F, x_gho];
            yy_gho_up_F = [yy_gho_up_F, y_gho];
        end
        
        % Generate the lower ghost particles (Solid)
        if (yy(i)>= yy_min-0*Dy/2-Dy/4)&&(yy(i)<=yy_min-0*Dy/2+Dy/4)&&(i_type(i)<20)
%         if (yy(i)>= yy_min+1*Dy/2-Dy/4)&&(yy(i)<=yy_min+1*Dy/2+Dy/4)&&(i_type(i)<20)
            i_ghost_low_S = i_ghost_low_S + 1;
            x_gho = xx(i)+Dx/2.;
%              x_gho = xx(i);
%             y_gho = yy_min - (j) * Dy/2.;
%             y_gho = yy_min - (j) * Dy/1. ;
%             if (j==1) 
                y_gho = yy_min - (j) * Dy/1. -2*Dy/2.;
%             else
%                 y_gho = yy_min - (j) * Dy/1.;
%             end
            xx_gho_low_S = [xx_gho_low_S, x_gho];
            yy_gho_low_S = [yy_gho_low_S, y_gho];
        end
        
%         %  Generate the right ghost particles (Solid)
%         if (xx(i)>= xx_max - Dx/2 - Dx/4) && (xx(i)<= xx_max - Dx/2 + Dx/4) && (yy(i)<=yb_max)
%             i_gho_right_S = i_gho_right_S + 1;
%             x_rep = xx(i) + (j+1) * Dx;
%             y_rep = yy(i);
%             xx_gho_right_S = [xx_gho_right_S, x_rep];
%             yy_gho_right_S = [yy_gho_right_S, y_rep];
%         end
        
    end
end

xx_temp_gho = [xx_temp_rep xx_gho_low_F xx_gho_low_S];
yy_temp_gho = [yy_temp_rep yy_gho_low_F yy_gho_low_S];
        

% for j = 1 : N_layer
%     for i = 1 : length(xx_temp_gho)
%         %  Generate the right ghost particles (Fluid)
% %         if (xx(i)>= xx_max - Dx/4) && (xx(i)<= xx_max + Dx/4) && (yy(i)<= max(yy_gho_right_S))
%         if (xx_temp_gho(i)>= xx_max - Dx/4) && (xx_temp_gho(i)<= xx_max + Dx/4) && (yy_temp_gho(i)<= max(yy_gho_right_S))
%             i_gho_right_F = i_gho_right_F + 1;
% %             x_rep = xx(i) + (j+1) * Dx - Dx/2;
% %             y_rep = yy(i);
%             x_rep = xx_temp_gho(i) + (j+1) * Dx - Dx/2;
%             y_rep = yy_temp_gho(i);
%             xx_gho_right_F = [xx_gho_right_F, x_rep];
%             yy_gho_right_F = [yy_gho_right_F, y_rep];
%         end
%     end
% end

i_gho_right = i_gho_right_S + i_gho_right_F;

Vcord.xx_rep_low_F = xx_rep_low_F;
Vcord.yy_rep_low_F = yy_rep_low_F;
Vcord.xx_rep_up_F = xx_rep_up_F;
Vcord.yy_rep_up_F = yy_rep_up_F;
Vcord.xx_rep_low_S = xx_rep_low_S;
Vcord.yy_rep_low_S = yy_rep_low_S;
% Vcord.xx_rep_right_S = xx_rep_right_S;
% Vcord.yy_rep_right_S = yy_rep_right_S;
% Vcord.xx_rep_right_F = xx_rep_right_F;
% Vcord.yy_rep_right_F = yy_rep_right_F;

Vcord.xx_gho_low_F = xx_gho_low_F;
Vcord.yy_gho_low_F = yy_gho_low_F;
Vcord.xx_gho_up_F = xx_gho_up_F;
Vcord.yy_gho_up_F = yy_gho_up_F;
Vcord.xx_gho_low_S = xx_gho_low_S;
Vcord.yy_gho_low_S = yy_gho_low_S;
% Vcord.xx_gho_right_S = xx_gho_right_S;
% Vcord.yy_gho_right_S = yy_gho_right_S;
% Vcord.xx_gho_right_F = xx_gho_right_F;
% Vcord.yy_gho_right_F = yy_gho_right_F;