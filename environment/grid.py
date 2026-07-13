#GridEnvironment class
from utils.libraries import *
from environment.rewards import rewards

class GridEnvironment:
    def __init__(self):
        #the grid is 24x16
        self.width = 24 
        self.height = 16
        self.safe_zone = [23,15]

        #Threat takes 2 steps for every 1 agent step
        self.threat_speed_multiplier = 2
    
    def reset(self):
        self.threat_pos = [0,0]
        self.agent_pos = [0, 0]
        while self.agent_pos == [0, 0]:
            self.agent_pos = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
            
        self.tokens = [] #Holds token coordinates
        self.collected_tokens = 0
        self.spawn_token(15) #initially 15 tokens in the grid
        
        self.rewards = rewards

        self.step_counter = 0


        return self.get_state()
        
    def get_state(self, is_chase=False):
        """Packages the environment state into a tuple format SARSA can read
        and informs the agent of nearby tokens and safe zone (not coordinatial, but directional)"""
        
        #1) Token Radar: Points to the nearest food
        if len(self.tokens) > 0:
            #find token with shortest Manhattan distance
            closest = min(self.tokens, key=lambda t: abs(t[0]-self.agent_pos[0]) + abs(t[1]-self.agent_pos[1]))
            token_dx = 1 if closest[0] > self.agent_pos[0] else (-1 if closest[0] < self.agent_pos[0] else 0)
            token_dy = 1 if closest[1] > self.agent_pos[1] else (-1 if closest[1] < self.agent_pos[1] else 0)
        else:
            token_dx, token_dy = 0, 0
        
        #2) Threat Radar: Points to the threat
        threat_dx = 1 if self.threat_pos[0] > self.agent_pos[0] else (-1 if self.threat_pos[0] < self.agent_pos[0] else 0)
        threat_dy = 1 if self.threat_pos[1] > self.agent_pos[1] else (-1 if self.threat_pos[1] < self.agent_pos[1] else 0)
        
        #3) Safe Zone Radar: Points to the bunker with dyamic masking
        if is_chase:
            safe_dx = 1 if self.safe_zone[0] > self.agent_pos[0] else (-1 if self.safe_zone[0] < self.agent_pos[0] else 0)
            safe_dy = 1 if self.safe_zone[1] > self.agent_pos[1] else (-1 if self.safe_zone[1] < self.agent_pos[1] else 0)
        else:
            #During foraging, blind the agent to the safe zone so it focuses on token
            safe_dx, safe_dy = 0, 0

        #4) An alarm bell (1 if true, 0 if false)
        chase_flag = 1 if is_chase else 0

        #Manhattan distance to threat
        threat_dist = abs(self.threat_pos[0] - self.agent_pos[0]) + abs(self.threat_pos[1] - self.agent_pos[1])
        #5) 1 if dangerously close (within 2 steps), 0 otherwise
        threat_close = 1 if threat_dist <= 2 else 0

        return (token_dx, token_dy, threat_dx, threat_dy, safe_dx, safe_dy, chase_flag, threat_close)

    def spawn_token(self, amount=1):
        """Spawn a specific amount of tokens in valid, empty coordinates."""
        for _ in range(amount):
            valid_position_found = False

            while not valid_position_found:
                new_pos = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
                #Valid positions:
                is_not_safe_zone = (new_pos != self.safe_zone)
                is_not_threat = (new_pos != self.threat_pos)
                is_not_already_token = (new_pos not in self.tokens)

                if is_not_safe_zone and is_not_threat and is_not_already_token:
                    self.tokens.append(new_pos)
                    valid_position_found = True

    def move_threat(self):
    #Basic manhattan distance pursuit
        if self.threat_pos[0] < self.agent_pos[0]:
            self.threat_pos[0] += 1
        elif self.threat_pos[0] > self.agent_pos[0]:
            self.threat_pos[0] -= 1
        elif self.threat_pos[1] < self.agent_pos[1]:
            self.threat_pos[1] += 1
        elif self.threat_pos[1] > self.agent_pos[1]:
            self.threat_pos[1] -= 1

    def step(self, agent_action, is_chase_phase=False):
        self.step_counter += 1
        
        # wait_and_act() pauses for 0.2 seconds per step. That means:
        # 1 turn/step of the game = 0.2 seconds
        # and 10 steps = 2 seconds
        if self.step_counter % 10 == 0: 
            self.spawn_token(3) #Spawn 3 new tokens every 2 seconds=every 10 steps
        
        reward = self.rewards["take_step"] #initialize step_reward, then keep adding
        
        old_safe_dist = abs(self.safe_zone[0] - self.agent_pos[0]) + abs(self.safe_zone[1] - self.agent_pos[1])

        #Agent position should be updated based on agent_action
        new_x = self.agent_pos[0]
        new_y = self.agent_pos[1]

        if agent_action == 0: new_y -= 1 #Up                
        elif agent_action == 1: new_x += 1 #Right                
        elif agent_action == 2: new_y += 1 #Down                
        elif agent_action == 3: new_x -= 1 #Left   
        # --NEW--
        elif agent_action == 4: pass #Stay   
        
        #Implement boundaries
        if 0<=new_x<self.width and 0<=new_y<self.height:
            self.agent_pos = [new_x,new_y] #Outsie the range, no position update
        else:
            reward += self.rewards["hit_wall"]

        #Token collection
        if self.agent_pos in self.tokens:
            self.tokens.remove(self.agent_pos)
            self.collected_tokens += 1
            reward += self.rewards["collect_token"]

        #Getting too close to the threat causes a penalty
        if not is_chase_phase and abs(self.threat_pos[0] - self.agent_pos[0]) + abs(self.threat_pos[1] - self.agent_pos[1]) <= 2:
            reward += self.rewards["near_threat"]

        new_safe_dist = abs(self.safe_zone[0] - self.agent_pos[0]) + abs(self.safe_zone[1] - self.agent_pos[1])


        if is_chase_phase:
            #Collision immediately after agent moves
            if self.agent_pos == self.threat_pos and self.agent_pos!=self.safe_zone:
                self.collected_tokens = 0 #agent loses all its tokens
                return self.get_state(is_chase_phase), "CAUGHT", self.rewards["is_caught"]

            if new_safe_dist < old_safe_dist:
                reward += self.rewards["approach_bunker"]  #Closer to bunker
            elif new_safe_dist > old_safe_dist:
                reward += self.rewards["leave_bunker"]  #Further from bunker

            if self.agent_pos == self.safe_zone:
                reward += self.rewards["hide_in_bunker"]

            #threat pursues
            for _ in range(self.threat_speed_multiplier):
                self.move_threat()

                #We check collision after EACH threat step
                if self.agent_pos == self.threat_pos and self.agent_pos!=self.safe_zone:
                    self.collected_tokens = 0 #agent loses all its tokens
                    return self.get_state(is_chase_phase), "CAUGHT", self.rewards["is_caught"]
        
        return self.get_state(is_chase_phase), "SAFE", reward

    def render(self, header_message=""):
        """Prints a visual of the grid to the terminal"""
        os.system('cls' if os.name == 'nt' else 'clear') #Clear terminal. cls for Windows, clear for Mac/Linux

        if header_message: #trial number, phase info... etc
            print(header_message)

        print("\n" + "="*50)
        for y in range(self.height): #0 to 15 y coordinates 
            row_string = ""
            for x in range(self.width): #0 to 23 x coordinates 
                current_pos = [x,y]

                #What exists at this specific coordinate?
                if current_pos == self.agent_pos:
                    row_string += "A " #agent
                elif current_pos == self.threat_pos:
                    row_string += "T " #threat
                elif current_pos == self.safe_zone:
                    row_string += "S " #safe zone
                elif current_pos in self.tokens:
                    row_string += "* " #token
                else:
                    row_string += ". " #empty space
            
            print(row_string)

        print(f"Tokens Collected: {self.collected_tokens}")
        print("="*50 + "\n")