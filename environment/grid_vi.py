from utils.libraries import *
from environment.rewards import rewards
from environment.grid import GridEnvironment

#Basic Inheritance
class GridVIEnvironment(GridEnvironment):
    def __init__(self):
        super().__init__() #Inherit
        self.width = 5
        self.height = 5
        self.safe_zone = [4,4]
        self.threat_speed_multiplier = 1 #1-to-1 movement ratio keeps the transition math clean

    def reset(self):
        super().reset()
        self.tokens = [self.tokens[0]] if self.tokens else []

    def get_state(self, is_chase=False):
        """Override to return absolute coordinates for VI."""
        t_x, t_y = self.tokens[0] if self.tokens else (-1,-1)
        return (self.agent_pos[0],self.agent_pos[1], 
                self.threat_pos[0],self.threat_pos[1],
                t_x, t_y)
    
    def get_all_states(self):
        """Generate every possible coordinate combination."""
        coords = [(x,y) for x in range(self.width) for y in range(self.height)]
        token_coords = coords + [(-1,-1)] #we define every possible state the token can occupy throughout the simulation

        # We take the Cartesian product of all coordinates
        # and build a list of all 16,250 possible game scenarios. 
        # The algorithm needs this list to sweep through and mathematically evaluate every situation offline
        return [
            (ax, ay, tx, ty, tokx, toky)
            for (ax, ay) in coords
            for (tx, ty) in coords
            for (tokx, toky) in token_coords
        ] #Agent positions, Threat positions, token positions

    def get_transition(self, state, action, is_chase_phase):
        """
        Simulates an action mathematically.
        Input: state tuple, action integer
        Returns: (next_state, reward, is_terminal_boolean)
        """
        ax, ay, tx, ty, tokx, toky = state
        reward = self.rewards["take_step"]
        is_terminal = False

        #Agent movement
        new_ax, new_ay = ax, ay
        if action == 0: new_ay -= 1 #Up                
        elif action == 1: new_ax += 1 #Right                
        elif action == 2: new_ay += 1 #Down                
        elif action == 3: new_ax -= 1 #Left   
        # action == 4 is Stay, coordinates do not change

        #Boundary check
        if 0<=new_ax<self.width and 0<=new_ay<self.height:
            pass #valid move
        else:
            new_ax, new_ay = ax, ay #Reset if wall hit
            reward += self.rewards["hit_wall"] 

        #Token collection
        new_tokx, new_toky = tokx, toky
        if new_ax == tokx and new_ay == toky:
            reward += self.rewards["collect_token"]
            new_tokx, new_toky = -1,-1 #we move collected token to the "graveyard=state outside the physical grid"

        new_tx, new_ty = tx,ty
        if is_chase_phase:
            old_safe_dist = abs(self.safe_zone[0] - ax) + abs(self.safe_zone[1] - ay)
            new_safe_dist = abs(self.safe_zone[0] - new_ax) + abs(self.safe_zone[1] - new_ay)

            if new_safe_dist < old_safe_dist:
                reward += self.rewards["approach_bunker"]  #Closer to bunker
            elif new_safe_dist > old_safe_dist:
                reward += self.rewards["leave_bunker"]  #Further from bunker

            #Bunker reward
            if [new_ax,new_ay] == self.safe_zone:
                reward += self.rewards["hide_in_bunker"]

            #Threat pursues
            if new_tx < new_ax: new_tx +=1
            elif new_tx > new_ax: new_tx -= 1
            elif new_ty < new_ay: new_ty += 1
            elif new_ty > new_ay: new_ty -= 1

            #Colllision check
            if new_ax == new_tx and new_ay == new_ty and [new_ax,new_ay]!=self.safe_zone:
                reward += self.rewards["is_caught"]
                is_terminal = True
                new_tokx, new_toky = -1,-1 #Lose tokens
        else:
            #Forage phase proximity penalt
            if abs(new_tx - new_ax) + abs(new_ty - new_ay) <= 2:
                reward += self.rewards["near_threat"]
        
        next_state = (new_ax,new_ay,new_tx,new_ty,new_tokx,new_toky)

        #Probability(=1 because env is deterministic),next state, reward, is terminal
        return [(1.0, next_state, reward, is_terminal)]
