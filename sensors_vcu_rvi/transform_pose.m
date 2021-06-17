%%%
% June 17th 2021, He Zhang fuyinzh@gmail.com 
% transform pose 
%

function new_pose = transform_pose(pose, rot, t)

    new_pose = zeros(size(pose));
    for i=1:size(pose,1)
        q = [pose(i, 7), pose(i, 4), pose(i, 5), pose(i, 6)]; 
        R = quat2rotm(q); 
        old_t = pose(i, 1:3); 
        new_R = rot * R; 
        new_t = rot * old_t' + t;      
        new_q = rotm2quat(new_R); 
        new_pose(i, :) = [new_t(1), new_t(2), new_t(3), new_q(2), new_q(3), new_q(4), new_q(1)]; 
        
    end

end