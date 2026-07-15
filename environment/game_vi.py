from environment.game import GameEnvironment
from environment.state import StateManager

class VIGameEnvironment(GameEnvironment):
    def __init__(self):
        super().__init__(width=5, height=5)
        self.safe_zone.x, self.safe_zone.y = 4, 4
        self.threat_speed_multiplier = 1

    def reset(self):
        super().reset()
        if self.tokens:
            self.tokens = [self.tokens[0]]
        return StateManager.get_vi_state(self)

    def get_all_states(self):
        coords = [(x, y) for x in range(self.width) for y in range(self.height)]
        token_coords = coords + [(-1, -1)]
        
        return [
            (ax, ay, tx, ty, tokx, toky)
            for (ax, ay) in coords
            for (tx, ty) in coords
            for (tokx, toky) in token_coords
        ]

    def get_transition(self, state, action, is_chase_phase):
        ax, ay, tx, ty, tokx, toky = state
        step_reward = self.rewards["take_step"]
        is_terminal = False

        new_ax, new_ay = ax, ay
        if action == 0: new_ay -= 1
        elif action == 1: new_ax += 1
        elif action == 2: new_ay += 1
        elif action == 3: new_ax -= 1

        if 0 <= new_ax < self.width and 0 <= new_ay < self.height:
            pass 
        else:
            new_ax, new_ay = ax, ay 
            step_reward += self.rewards["hit_wall"] 

        new_tokx, new_toky = tokx, toky
        if new_ax == tokx and new_ay == toky:
            step_reward += self.rewards["collect_token"]
            new_tokx, new_toky = -1, -1 

        new_tx, new_ty = tx, ty
        if is_chase_phase:
            old_safe_dist = abs(self.safe_zone.x - ax) + abs(self.safe_zone.y - ay)
            new_safe_dist = abs(self.safe_zone.x - new_ax) + abs(self.safe_zone.y - new_ay)

            if new_safe_dist < old_safe_dist:
                step_reward += self.rewards["approach_bunker"]  
            elif new_safe_dist > old_safe_dist:
                step_reward += self.rewards["leave_bunker"]  

            if new_ax == self.safe_zone.x and new_ay == self.safe_zone.y:
                step_reward += self.rewards["hide_in_bunker"]

            if new_tx < new_ax: new_tx += 1
            elif new_tx > new_ax: new_tx -= 1
            elif new_ty < new_ay: new_ty += 1
            elif new_ty > new_ay: new_ty -= 1

            if new_ax == new_tx and new_ay == new_ty and (new_ax, new_ay) != (self.safe_zone.x, self.safe_zone.y):
                step_reward += self.rewards["is_caught"]
                is_terminal = True
                new_tokx, new_toky = -1, -1 
        else:
            if abs(new_tx - new_ax) + abs(new_ty - new_ay) <= 2:
                step_reward += self.rewards["near_threat"]
        
        next_state = (new_ax, new_ay, new_tx, new_ty, new_tokx, new_toky)
        return [(1.0, next_state, step_reward, is_terminal)]