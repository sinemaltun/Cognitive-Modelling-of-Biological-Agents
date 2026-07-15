import random
from entities.player import Player
from entities.predator import Predator
from entities.token import Token
from entities.safezone import SafeZone
from config.rewards import rewards
from environment.state import StateManager

class GameEnvironment:
    def __init__(self, width=24, height=16):
        self.width = width
        self.height = height
        self.safe_zone = SafeZone(width - 1, height - 1)
        self.tokens = []
        self.step_counter = 0
        self.threat_speed_multiplier = 2
        self.rewards = rewards
        
    def reset(self):
        self.player = Player(random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        self.predator = Predator(0, 0)
        self.tokens.clear()
        self.spawn_tokens(15)
        self.step_counter = 0
        return StateManager.get_sarsa_state(self)
        
    def spawn_tokens(self, amount):
        for _ in range(amount):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.tokens.append(Token(x, y))

    def step(self, agent_action, is_chase_phase=False):
        self.step_counter += 1
        if self.step_counter % 10 == 0:
            self.spawn_tokens(3)

        step_reward = self.rewards["take_step"]
        old_safe_dist = abs(self.safe_zone.x - self.player.x) + abs(self.safe_zone.y - self.player.y)

        dx, dy = 0, 0
        if agent_action == 0: dy = -1
        elif agent_action == 1: dx = 1
        elif agent_action == 2: dy = 1
        elif agent_action == 3: dx = -1

        new_x = self.player.x + dx
        new_y = self.player.y + dy

        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            self.player.move(dx, dy)
        else:
            step_reward += self.rewards["hit_wall"]

        collected = [t for t in self.tokens if t.x == self.player.x and t.y == self.player.y]
        if collected:
            for t in collected:
                self.tokens.remove(t)
            self.player.tokens += len(collected)
            step_reward += self.rewards["collect_token"]

        if not is_chase_phase and abs(self.predator.x - self.player.x) + abs(self.predator.y - self.player.y) <= 2:
            step_reward += self.rewards["near_threat"]

        new_safe_dist = abs(self.safe_zone.x - self.player.x) + abs(self.safe_zone.y - self.player.y)

        if is_chase_phase:
            if self.player.get_pos() == self.predator.get_pos() and self.player.get_pos() != self.safe_zone.get_pos():
                self.player.tokens = 0
                return StateManager.get_sarsa_state(self, is_chase_phase), "CAUGHT", self.rewards["is_caught"]

            if new_safe_dist < old_safe_dist:
                step_reward += self.rewards["approach_bunker"]
            elif new_safe_dist > old_safe_dist:
                step_reward += self.rewards["leave_bunker"]

            if self.player.get_pos() == self.safe_zone.get_pos():
                step_reward += self.rewards["hide_in_bunker"]

            for _ in range(self.threat_speed_multiplier):
                self.predator.move_towards(self.player.x, self.player.y)
                if self.player.get_pos() == self.predator.get_pos() and self.player.get_pos() != self.safe_zone.get_pos():
                    self.player.tokens = 0
                    return StateManager.get_sarsa_state(self, is_chase_phase), "CAUGHT", self.rewards["is_caught"]

        return StateManager.get_sarsa_state(self, is_chase_phase), "SAFE", step_reward