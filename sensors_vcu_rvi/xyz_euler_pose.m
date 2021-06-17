%%%
% June 17th 2021, He Zhang fuyinzh@gmail.com 
% compute xyz and rpy from trajectory 
%

function [xyz, rpy] = xyz_euler_pose(pose)

    xyz = pose(:,1:3); 
    rpy = zeros(size(pose,1), 3); 
    
    for i=1:size(pose,1)
        q = [pose(i, 7), pose(i, 4), pose(i, 5), pose(i, 6)]; 
        R = quat2rotm(q); 
        % fprintf('i=%d\n', i);
        e = rotm2eul(R); 
        % e = R2e(R);
        rpy(i, :) = rad2deg(e); 
    end
end