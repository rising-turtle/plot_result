
%
% Nov. 3, 2018, He Zhang, hzhang8@vcu.edu 
% 
% plot the trajectories of result in comp_d6 folder 
%

addpath('../../');
addpath('.');

tr1 = dlmread('./comp_d4/d4_vins_mono.csv'); 
tr2 = dlmread('./comp_d4/d4_vins_rgbd_f230.csv'); 
tr3 = dlmread('./comp_d4/d4_dvio_no_ini.csv'); 
tr4 = dlmread('./comp_d4/d4_dvio_w_ini.csv'); 


plot_xyz(tr1(:,2), tr1(:,3), tr1(:,4), 'r-');
hold on;
plot_xyz(tr2(:,2), tr2(:,3), tr2(:,4), 'c-');
hold on;
plot_xyz(tr3(:,2), tr3(:,3), tr3(:,4), 'g-'); 
hold on;
plot_xyz(tr4(:,2), tr4(:,3), tr4(:,4), 'b-'); 
hold on;
grid on
plot3(0, 20.0, 0, 'k*');
legend('VINS-Mono', 'VINS-RGBD', 'DVIO', 'DVIO w Initial'); 




