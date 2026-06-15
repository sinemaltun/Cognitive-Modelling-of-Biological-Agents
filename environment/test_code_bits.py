#A script to test small bits of my code

from libraries import *

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