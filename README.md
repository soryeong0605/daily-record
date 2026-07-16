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






0.0089, +0.0000, +0.0000], safety_net_hits=132
[FrankaController] age=0.015, target=[+0.0000, +0.0000, +0.0000, +0.0195], send=[+0.0000, +0.0000, +0.0000, +0.0032], safety_net_hits=137
[FrankaController] age=0.016, target=[+0.0141, +0.0592, +0.0000, +0.0519], send=[+0.0002, +0.0003, +0.0000, +0.0010], safety_net_hits=141
[FrankaController] age=0.001, target=[+0.0000, -0.0242, +0.0000, +0.0097], send=[+0.0000, -0.0070, +0.0000, +0.0097], safety_net_hits=146
[FrankaController] Native velocity loop exception: libfranka: Move command aborted: motion aborted by reflex! ["cartesian_motion_generator_joint_acceleration_discontinuity"]
[FrankaController] Force stop requested
Exception in thread Thread-1 (_velocity_loop):
Traceback (most recent call last):
  File "/home/holabrb/miniforge3/envs/franka12/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
    self.run()
  File "/home/holabrb/miniforge3/envs/franka12/lib/python3.12/threading.py", line 1012, in run
    self._target(*self._args, **self._kwargs)
  File "/home/holabrb/woonjoo/Jul16/cosine_visual/franka_cosine.py", line 447, in _velocity_loop
    state, duration = active.readOnce()
                      ^^^^^^^^^^^^^^^^^
pylibfranka._pylibfranka.ControlException: libfranka: Move command aborted: motion aborted by reflex! ["cartesian_motion_generator_joint_acceleration_discontinuity"]
[Main] Live monitor window opened. Close it or Ctrl+C to stop.
[Main] Teleop running with hybrid cosine/slew-rate smoother
[Main] Press Ctrl+C to stop
[Main] Franka velocity thread died.
[FrankaController] Native Cartesian velocity servo stopped
[Main] Stopped
(franka12) holabrb@franka:~/woonjoo/Jul16/cosine_visual$
