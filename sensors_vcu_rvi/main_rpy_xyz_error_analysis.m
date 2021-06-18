%%%
% June 17th 2021, He Zhang fuyinzh@gmail.com 
% transform trajectories to obtain rpy and xyz error analysis 
%

%% load data

gt = load('./data/lab_motion2/ground_truth.csv'); 
vins_mono = load('./data/lab_motion2/VINS-Mono.csv'); 
vins_rgbd = load('./data/lab_motion2/VINS-RGBD.csv'); 
dui_vio = load('./data/lab_motion2/DUI-VIO.csv'); 

t_gt = gt(:,1); 
t_mono = vins_mono(:,1); 
t_rgbd = vins_rgbd(:,1); 
t_dui = dui_vio(:,1); 

[xyz_gt, rpy_gt] = xyz_euler_pose(gt(:, 2:end)); 

%% transform trajectory 
n = 250; % number of frames to align trajectories 
vins_mono = align_pose(t_gt, xyz_gt, vins_mono, n);
vins_rgbd = align_pose(t_gt, xyz_gt, vins_rgbd, n); 
dui_vio = align_pose(t_gt, xyz_gt, dui_vio, n); 

%% get rpy xyz 
[xyz_mono, rpy_mono] = xyz_euler_pose(vins_mono); 
[xyz_rgbd, rpy_rgbd] = xyz_euler_pose(vins_rgbd);
[xyz_dui, rpy_dui] = xyz_euler_pose(dui_vio); 

%% plot results 
linewidth = 1.2;
font_size = 14;
t_max = t_dui(end) + 0.5;

subplot(3, 2, 1); 
plot(t_gt, xyz_gt(:,1), 'k--', 'LineWidth', linewidth); 
hold on; 
plot(t_mono, xyz_mono(:,1), 'b-', 'LineWidth', linewidth); 
plot(t_rgbd, xyz_rgbd(:,1), 'g-', 'LineWidth', linewidth); 
plot(t_dui, xyz_dui(:,1), 'r-', 'LineWidth', linewidth); 
xlim([0, t_max]); 
xlabel('timestamp (s)', 'FontSize', font_size); 
ylabel('x (m)', 'FontSize', font_size);
legend('groundtruth', 'VINS-Mono', 'VINS-RGBD', 'DUI-VIO', 'FontSize', 10);

subplot(3,2,2); 
plot(t_gt, rpy_gt(:,1), 'k--', 'LineWidth', linewidth); 
hold on; 
plot(t_mono, rpy_mono(:,1), 'b-', 'LineWidth', linewidth);
plot(t_rgbd, rpy_rgbd(:,1), 'g-', 'LineWidth', linewidth); 
plot(t_dui, rpy_dui(:,1), 'r-', 'LineWidth', linewidth); 
xlim([0, t_max]); 
xlabel('timestamp (s)', 'FontSize', font_size); 
ylabel('roll (degree)', 'FontSize', font_size);

subplot(3, 2, 3); 

plot(t_gt, xyz_gt(:,2), 'k--', 'LineWidth', linewidth); 
hold on; 
plot(t_mono, xyz_mono(:,2), 'b-', 'LineWidth', linewidth); 
plot(t_rgbd, xyz_rgbd(:,2), 'g-', 'LineWidth', linewidth); 
plot(t_dui, xyz_dui(:,2), 'r-', 'LineWidth', linewidth); 
xlim([0, t_max]); 
xlabel('timestamp (s)', 'FontSize', font_size); 
ylabel('y (m)', 'FontSize', font_size);

subplot(3,2,4); 
plot(t_gt, rpy_gt(:,2), 'k--', 'LineWidth', linewidth); 
hold on; 
plot(t_mono, rpy_mono(:,2), 'b-', 'LineWidth', linewidth);
plot(t_rgbd, rpy_rgbd(:,2), 'g-', 'LineWidth', linewidth); 
plot(t_dui, rpy_dui(:,2), 'r-', 'LineWidth', linewidth); 
xlim([0, t_max]); 
xlabel('timestamp (s)', 'FontSize', font_size); 
ylabel('pitch (degree)', 'FontSize', font_size);

subplot(3, 2, 5); 
plot(t_gt, xyz_gt(:,3), 'k--', 'LineWidth', linewidth); 
hold on; 
plot(t_mono, xyz_mono(:,3), 'b-', 'LineWidth', linewidth); 
plot(t_rgbd, xyz_rgbd(:,3), 'g-', 'LineWidth', linewidth); 
plot(t_dui, xyz_dui(:,3), 'r-', 'LineWidth', linewidth); 
xlim([0, t_max]); 
xlabel('timestamp (s)', 'FontSize', font_size); 
ylabel('z (m)', 'FontSize', font_size); 

subplot(3,2,6); 
plot(t_gt, rpy_gt(:,3), 'k--', 'LineWidth', linewidth); 
hold on; 
plot(t_mono, rpy_mono(:,3), 'b-', 'LineWidth', linewidth);
plot(t_rgbd, rpy_rgbd(:,3), 'g-', 'LineWidth', linewidth); 
plot(t_dui, rpy_dui(:,3), 'r-', 'LineWidth', linewidth); 
xlim([0, t_max]); 
xlabel('timestamp (s)', 'FontSize', font_size); 
ylabel('yaw (degree)', 'FontSize', font_size);








