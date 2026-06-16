#A script to test small bits of my code

from libraries import *
"""
#the grid is 24x16
agent_pos = [0, 0]
while agent_pos == [0, 0]:
    agent_pos = [random.randint(0, 4), random.randint(0, 4)]
threat_pos = [0,0]
safe_zone = [4,4]
tokens = []
total_tokens = 5 #initially 11 tokens in the grid
token_count = 0
forbidden_coordinates = [threat_pos, safe_zone]

while token_count != total_tokens:  
    position = [random.randint(0, 4), random.randint(0, 4)]
    print(position)
    if position in forbidden_coordinates:
        pass
    else:
        forbidden_coordinates.append(position)
        tokens.append(position)
        token_count += 1

print("\nToken Coordinates:",tokens)
"""


agent_pos = [2,3]
threat_pos = [0,9]
safe_zone = [23,15]
tokens = [[1,2],[8,6]]

"""Prints a visual of the grid to the terminal"""
os.system('cls' if os.name == 'nt' else 'clear')
print("\n" + "="*50)
for y in range(16): #0 to 14 y coordinates 
    row_string = ""
    for x in range(24): #0 to 23 x coordinates 
        current_pos = [x,y]

        #What exists at this specific coordinate?
        if current_pos == agent_pos:
            row_string += "A " #agent
        elif current_pos == threat_pos:
            row_string += "T " #threat
        elif current_pos == safe_zone:
            row_string += "S " #safe zone
        elif current_pos in tokens:
            row_string += "* " #token
        else:
            row_string += ". " #empty space
    
        print(row_string)
    print("="*50 + "\n")