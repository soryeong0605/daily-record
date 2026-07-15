# daily-record
기록용용용
안녕
function v_ref = smooth_accel(v_target)
  
   
    persistent v_prev;     
    dt = 0.002;           
    max_accel = 30;       
    % ---------------------------------

    if isempty(v_prev)
        v_prev = 0;
    end

   
    diff = v_target - v_prev;
    
    max_step = max_accel * dt;

    if abs(diff) < max_step
      
        v_ref = v_target;
    else
      
        v_ref = v_prev + sign(diff) * max_step;
    end

    v_prev = v_ref;
end
