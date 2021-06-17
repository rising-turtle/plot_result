%%%
% June 17th 2021, He Zhang fuyinzh@gmail.com 
% find matched pose by timestamp  
%

function index = find_matched_by_timestamp(time, gt_time, n)
    
    
    index = [];
    j = 1; 
    n = min(n, size(gt_time,1));
    for t = 1:n
        cur_t = time(t); 
        
        while j < size(gt_time,1)
            
            if(abs(gt_time(j,1) - cur_t) < (1/120.))
                index(end+1) = j; 
                break;
            end
            
            % TODO: remove this point, but now just assign a close one 
            % which does not affect the results too much 
            if(gt_time(j,1) - cur_t > 1/120.)
                index(end+1) = j; 
                break;
            end
            
            j= j+1; 
        end
    end
end