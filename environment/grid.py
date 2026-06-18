#GridEnvironment class

# --- TO DOs ---
# 1. Implement boundary logics --done
# 2. Episode reset: Start new trial --done
# 3. Define reward returns, step() function should retur alongside game status. --done
# 4. Package environment state into a tuple format for SARSA --done
# 5. ASCII Visualizations --done
# 6. Add action "stay in place" --done

from utils.libraries import *

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
            self.agent_pos = [random.randint(0, 23), random.randint(0, 15)]
            
        self.tokens = [] #Holds token coordinates
        self.collected_tokens = 0
        self.spawn_token(15) #initially 15 tokens in the grid

        self.rewards = {"take_step": -1,
                        "hit_wall": -2,
                        "collect_token": +10,
                        "is_caught": -100}
        
        self.step_counter = 0

        return self.get_state()
        
    def get_state(self):
        """Packages the environment state into a tuple format SARSA can read."""
        return (self.agent_pos[0], self.agent_pos[1],
                self.threat_pos[0],self.threat_pos[1])

    def spawn_token(self, amount=1):
        """Spawn a specific amount of tokens in valid, empty coordinates."""
        for _ in range(amount):
            valid_position_found = False

            while not valid_position_found:
                new_pos = [random.randint(0, 23), random.randint(0, 15)]
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
        
        step_reward = self.rewards["take_step"] #initialize step_reward, then keep adding

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
        if 0<=new_x<=23 and 0<=new_y<=15:
            self.agent_pos = [new_x,new_y] #Outsie the range, no position update
        else:
            step_reward += self.rewards["hit_wall"]

        #Token collection
        if self.agent_pos in self.tokens:
            self.tokens.remove(self.agent_pos)
            self.collected_tokens += 1
            step_reward += self.rewards["collect_token"]

        #Collision immediately after agent moves
        if self.agent_pos == self.threat_pos and self.agent_pos!=self.safe_zone:
            self.collected_tokens = 0 #agent loses all its tokens
            return self.get_state(), "CAUGHT", self.rewards["is_caught"]

        if is_chase_phase:
            for _ in range(self.threat_speed_multiplier):
                self.move_threat() #threat pursues

                #We check collision after EACH threat step
                if self.agent_pos == self.threat_pos and self.agent_pos!=self.safe_zone:
                    self.collected_tokens = 0 #agent loses all its tokens
                    return self.get_state(), "CAUGHT", self.rewards["is_caught"]
        
        return self.get_state(), "SAFE", step_reward

    def render(self, header_message=""):
        """Prints a visual of the grid to the terminal"""
        os.system('cls' if os.name == 'nt' else 'clear') #Clear terminal. cls for Windows, clear for Mac/Linux

        if header_message: #trial number, phase info... etc
            print(header_message)

        print("\n" + "="*50)
        for y in range(16): #0 to 15 y coordinates 
            row_string = ""
            for x in range(24): #0 to 23 x coordinates 
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