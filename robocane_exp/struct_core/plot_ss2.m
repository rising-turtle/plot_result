
%
% Dec. 28, 2020, He Zhang, fuyinzh@gmail.com  
% 
% plot the trajectories of result in robocane_ss2 folder 
%
% D5-lab2, D8-lab6

addpath('../../');
addpath('.');

tr1 = dlmread('./robocane_ss2/gt_comp/lab03_gt_cp.csv');  % lab02_gt_cp
tr2 = dlmread('./robocane_ss2/vins-mono/lab_3_tum_cp.csv'); % lab_2_tum_cp
tr3 = dlmread('./robocane_ss2/vins_rgbd/lab_3_vins_cp.csv'); % lab_2_vins_cp
tr4 = dlmread('./robocane_ss2/dvio_3/lab_3_tum_cp.csv'); % lab_2_tum_cp


plot_xyz(tr1(:,2), tr1(:,3), tr1(:,4), 'k-');
hold on;

plot_xyz(tr2(:,2), tr2(:,3), tr2(:,4), 'b-');
hold on;
plot_xyz(tr3(:,2), tr3(:,3), tr3(:,4), 'g-'); 
hold on;
plot_xyz(tr4(:,2), tr4(:,3), tr4(:,4), 'r-'); 
hold on;
grid on; 

plot3(tr1(1,2), tr1(1,3), tr1(1,4), 'ko', 'MarkerSize', 12);
hold on;
plot3(tr1(end,2), tr1(end,3), tr1(end,4), 'kx', 'MarkerSize', 12);
hold on;
plot3(tr2(end,2), tr2(end,3), tr2(end,4), 'bx', 'MarkerSize', 12);
hold on;
plot3(tr3(end,2), tr3(end,3), tr3(end,4), 'gx', 'MarkerSize', 12);
hold on;
plot3(tr4(end,2), tr4(end,3), tr4(end,4), 'rx', 'MarkerSize', 12);

legend('Ground Truth', 'VINS-Mono', 'VINS-RGBD', 'DVIO', 'Start point', 'End point', ... 
    'End point of VINS-Mono', 'End point of VINS-RGBD', 'End point of DVIO'); 


