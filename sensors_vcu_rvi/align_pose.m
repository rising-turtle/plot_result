%%%
% June 17th 2021, He Zhang fuyinzh@gmail.com 
% align estimated trajectory to ground truth 
%

function new_pose = align_pose(t_gt, xyz_gt, pose_est, n)
    
    %% find index with the matched timestamp 
    
    index = find_matched_by_timestamp(pose_est(:,1), t_gt, n); 
    
    gt_pt = xyz_gt(index, :); 
    et_pt = pose_est(1:n, 2:4); 
    
    %% compute transformation 
    [rot, t] = eq_point(gt_pt', et_pt');
    
    %% transform pose 
    new_pose = transform_pose(pose_est(:, 2:end), rot, t); 
    
end