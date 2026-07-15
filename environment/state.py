class StateManager:
    @staticmethod
    def get_sarsa_state(game_env, is_chase=False):
        player_pos = game_env.player.get_pos()
        threat_pos = game_env.predator.get_pos()
        
        if game_env.tokens:
            closest = min(game_env.tokens, key=lambda t: abs(t.x - player_pos[0]) + abs(t.y - player_pos[1]))
            token_dx = 1 if closest.x > player_pos[0] else (-1 if closest.x < player_pos[0] else 0)
            token_dy = 1 if closest.y > player_pos[1] else (-1 if closest.y < player_pos[1] else 0)
        else:
            token_dx, token_dy = 0, 0
            
        threat_dx = 1 if threat_pos[0] > player_pos[0] else (-1 if threat_pos[0] < player_pos[0] else 0)
        threat_dy = 1 if threat_pos[1] > player_pos[1] else (-1 if threat_pos[1] < player_pos[1] else 0)
        
        if is_chase:
            safe_pos = game_env.safe_zone.get_pos()
            safe_dx = 1 if safe_pos[0] > player_pos[0] else (-1 if safe_pos[0] < player_pos[0] else 0)
            safe_dy = 1 if safe_pos[1] > player_pos[1] else (-1 if safe_pos[1] < player_pos[1] else 0)
        else:
            safe_dx, safe_dy = 0, 0

        chase_flag = 1 if is_chase else 0
        threat_dist = abs(threat_pos[0] - player_pos[0]) + abs(threat_pos[1] - player_pos[1])
        threat_close = 1 if threat_dist <= 2 else 0

        return (token_dx, token_dy, threat_dx, threat_dy, safe_dx, safe_dy, chase_flag, threat_close)
        
    @staticmethod
    def get_vi_state(game_env):
        player_pos = game_env.player.get_pos()
        threat_pos = game_env.predator.get_pos()
        t_x, t_y = game_env.tokens[0].get_pos() if game_env.tokens else (-1, -1)
        
        return (player_pos[0], player_pos[1], threat_pos[0], threat_pos[1], t_x, t_y)