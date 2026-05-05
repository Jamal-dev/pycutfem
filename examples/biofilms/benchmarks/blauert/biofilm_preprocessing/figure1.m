function figure1
%
% This function returns the obtained coordinates of the control points in
% the figure
%
% Coordinates of the four control nodes of the figure domain are saved in
% "domain.txt" .
% Coordinates of the control nodes at the biofilm interface are saved in
% "biofilm.txt".
%
% Copyright @ Dianlei Feng

% I = imread('Basic_t=2_INK_RG.png');
I = imread('rule.png');
imshow(I)

[x1,y1] = ginput(4);
A1 = [x1, y1];
fileID = fopen('domain.txt','w');
fprintf(fileID,'%12s %12s\n','x','y');
fprintf(fileID,'%12.8f %12.8f\n',A1');
fclose(fileID);

% x1
% y1
% plot(x1,y1)

pause(5)

[x2,y2] = ginput;
% x2 
% y2
A2 = [x2, y2];
fileID = fopen('biofilm.txt','w');
fprintf(fileID,'%12s %12s\n','x','y');
fprintf(fileID,'%12.8f %12.8f\n',A2');
fclose(fileID);
plot(x2,y2)