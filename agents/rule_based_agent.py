import numpy as np

# class RuleBasedShepherd:
#     def __init__(self, env_init, drive_distance=50):
#         """
#         drive_distance: how far behind the furthest sheep to drive
#         """
#         self.drive_distance = drive_distance
#         self.env = env_init

#     def _check_goal(self):
#         # Uses the environment's internal state to see which sheep are finished
#         reached = np.zeros(len(self.env.sheep), dtype=bool)
#         for i, s in enumerate(self.env.sheep):
#             for g in self.env.goals:
#                 if np.linalg.norm(s.pos - g) < self.env.goal_radius:
#                     reached[i] = True
#         return reached

#     def act(self, obs, n_sheep, n_shepherds=1, shepherd_idx=0):
#         """
#         obs: flattened observation array
#         n_sheep: number of sheep
#         n_shepherds: number of shepherds
#         """
#         # 1. FIX: Define how many values each object has in the observation array
#         # In the differential drive env: Sheep=2 (x,y), Shepherd=3 (x,y,theta), Goal=2 (x,y)
#         SHEP_DIM = 3 
        
#         # 2. Extract Sheep Positions
#         sheep_pos = obs[:2*n_sheep].reshape((n_sheep, 2))
        
#         # 3. Extract Shepherd Position (We only need X and Y for logic)
#         shep_start = 2 * n_sheep + (SHEP_DIM * shepherd_idx)
#         shepherd_pos = obs[shep_start : shep_start + 2] # Only take x, y
#         # (The 3rd value, obs[shep_start + 2], is theta, which we ignore for this heuristic)

#         # 4. Extract Goal Positions
#         # Goals start after ALL sheep and ALL shepherds
#         goals_start = 2 * n_sheep + (SHEP_DIM * n_shepherds)
#         goals_data = obs[goals_start:]
        
#         # Determine number of goals based on remaining data
#         n_goals = len(goals_data) // 2
#         goals_pos = goals_data.reshape((n_goals, 2))
        
#         # Assign goal (Level 4 has 2 goals, Levels 1-3 usually have 1)
#         goal = goals_pos[shepherd_idx] if len(goals_pos) > shepherd_idx else goals_pos[0]

#         # 5. Heuristic Logic: Find furthest sheep that isn't safe yet
#         reached = self._check_goal()
#         distances = np.linalg.norm(sheep_pos - goal, axis=1)
        
#         # Mask out reached sheep by setting their distance to -1
#         distances[reached] = -1
        
#         if np.all(reached):
#             return np.array([0.0, 0.0]) # Everything is done

#         idx = np.argmax(distances)
#         furthest_sheep = sheep_pos[idx]

#         # 6. Calculate Driving Point (Goal -> Sheep -> DrivePoint)
#         g2s = furthest_sheep - goal
#         norm = np.linalg.norm(g2s)
#         if norm < 1e-6:
#             g2s = np.array([1.0, 0.0])
#         else:
#             g2s /= norm

#         drive_point = furthest_sheep + g2s * self.drive_distance
#         drive_point = np.clip(drive_point, 10, self.env.screen_size - 10)

#         # 7. Compute vector toward driving point
#         vec = drive_point - shepherd_pos
        
#         # If we are basically there, stop
#         if np.linalg.norm(vec) < 5:
#             return np.array([0.0, 0.0])

#         # Normalize the vector to get a direction
#         direction = vec / (np.linalg.norm(vec) + 1e-6)
        
#         # IMPORTANT: This returns an XY movement vector.
#         # If your environment expects [LeftWheel, RightWheel], 
#         # you need to convert this vector to wheel speeds:
#         return self._vector_to_wheels(direction, obs[shep_start + 2])

#     def _vector_to_wheels(self, target_dir, current_theta):
#         """
#         Converts a target direction vector into [LeftWheel, RightWheel] actions
#         for a differential drive robot.
#         """
#         target_angle = np.arctan2(target_dir[1], target_dir[0])
#         angle_diff = target_angle - current_theta
        
#         # Normalize angle to [-pi, pi]
#         angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        
#         # Simple PD-like control for wheels
#         forward_speed = 0.8 if abs(angle_diff) < np.pi/2 else 0.1
#         rotation_speed = angle_diff * 0.5
        
#         left_wheel = forward_speed - rotation_speed
#         right_wheel = forward_speed + rotation_speed
        
#         return np.clip([left_wheel, right_wheel], -1, 1)

class RuleBasedShepherd:
    def __init__(self, env_init,drive_distance=50):
        """
        drive_distance: how far behind the furthest sheep to drive
        """
        self.drive_distance = drive_distance
        self.env=env_init

    def _check_goal(self):
        reached = np.zeros(len(self.env.sheep), dtype=bool)
        for i, s in enumerate(self.env.sheep):
            for g in self.env.goals:
                if np.linalg.norm(s.pos - g) < self.env.goal_radius:
                    reached[i] = True
        return reached

    def act(self, obs, n_sheep, n_shepherds=1, shepherd_idx=0):
        """
        obs: flattened observation array
        n_sheep: number of sheep
        n_shepherds: number of shepherds
        shepherd_idx: index of this shepherd
        """
        # Extract positions
        sheep_pos = obs[:2*n_sheep].reshape((n_sheep,2))
        print(sheep_pos.shape)
        shepherd_pos = obs[2*n_sheep + 2*shepherd_idx : 2*n_sheep + 2*(shepherd_idx+1)]
        print(shepherd_pos.shape)
        print(obs.shape)
        goals_pos = obs[2*n_sheep + 2*n_shepherds:].reshape((n_shepherds,2))
        goal = goals_pos[shepherd_idx]

        # Furthest sheep from goal
        distances = np.linalg.norm(sheep_pos - goal, axis=1)
        idx = np.argmax(distances*(~self._check_goal()))
        furthest_sheep = sheep_pos[idx]

        # Line from goal → sheep
        g2s = furthest_sheep - goal
        norm = np.linalg.norm(g2s)
        if norm < 1e-6:
            g2s = np.array([1.0,0.0])
        else:
            g2s /= norm

        # Driving point beyond sheep
        drive_point = furthest_sheep + g2s * self.drive_distance
        drive_point=np.clip(drive_point,0,self.env.screen_size)

        # Compute vector toward driving point
        vec = drive_point - shepherd_pos

        return np.array(vec)/sum(abs(vec))
