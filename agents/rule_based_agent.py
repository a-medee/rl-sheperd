import numpy as np

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
        shepherd_pos = obs[2*n_sheep + 2*shepherd_idx : 2*n_sheep + 2*(shepherd_idx+1)]
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
        # print(f"Shepherd {shepherd_idx} driving to point {drive_point} from pos {shepherd_pos} with vector {vec}")

        return np.array(vec)/sum(abs(vec))
