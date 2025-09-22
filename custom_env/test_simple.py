#!/usr/bin/env python3
"""
Test simple de l'environnement sans gymnasium.
"""

import numpy as np
from collections import deque

class SimpleRacingEnv:
    def __init__(self):
        self.window_horizon = 10
        self.car_position_in_window = 3
        self.car_y = 1.0
        self.car_vy = 0.0
        self.step_count = 0
        self.road_history = deque(maxlen=self.window_horizon)
        
    def generate_road_segment(self):
        """Génère un segment de route avec bords haut et bas de façon aléatoire."""
        # Bord haut (0 à 1) - position aléatoire
        top_border = np.random.uniform(0.1, 0.9)
        
        # Bord bas (0 à 1) - position aléatoire  
        bottom_border = np.random.uniform(0.1, 0.9)
        
        # Vérifier que la zone libre existe (top_border < 2 - bottom_border)
        if top_border >= 2.0 - bottom_border:
            # Ajuster pour créer une zone libre
            if top_border < bottom_border:
                bottom_border = 2.0 - top_border - 0.1
            else:
                top_border = 2.0 - bottom_border - 0.1
        
        # Clamper les valeurs finales
        top_border = np.clip(top_border, 0.1, 0.9)
        bottom_border = np.clip(bottom_border, 0.1, 0.9)
        
        return top_border, bottom_border
    
    def reset(self):
        """Reset l'environnement."""
        self.car_y = 1.0
        self.car_vy = 0.0
        self.step_count = 0
        self.road_history.clear()
        
        # Générer l'historique initial
        for i in range(self.window_horizon):
            top_border, bottom_border = self.generate_road_segment()
            self.road_history.append((top_border, bottom_border))
        
        return self.get_observation()
    
    def get_observation(self):
        """Retourne l'observation actuelle."""
        obs = [self.car_y, self.car_vy]
        for top_border, bottom_border in self.road_history:
            obs.extend([top_border, bottom_border])
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """Exécute une action."""
        self.step_count += 1
        
        # Appliquer l'action
        if action == 0:  # Descendre
            self.car_vy = max(-0.1, self.car_vy - 0.02)
        elif action == 1:  # Rester
            self.car_vy *= 0.9
        elif action == 2:  # Monter
            self.car_vy = min(0.1, self.car_vy + 0.02)
        
        # Mettre à jour la position
        self.car_y += self.car_vy
        self.car_y = np.clip(self.car_y, 0.0, 2.0)
        
        # Faire défiler la route
        self.road_history.popleft()
        top_border, bottom_border = self.generate_road_segment()
        self.road_history.append((top_border, bottom_border))
        
        # Calculer la récompense
        current_borders = list(self.road_history)[self.car_position_in_window]
        top_border, bottom_border = current_borders
        free_zone_top = top_border
        free_zone_bottom = 2.0 - bottom_border
        
        if free_zone_top <= self.car_y <= free_zone_bottom:
            reward = 1.0
            terminated = False
        else:
            reward = -10.0
            terminated = True
        
        return self.get_observation(), reward, terminated, False, {}
    
    def print_state(self):
        """Affiche l'état actuel."""
        current_borders = list(self.road_history)[self.car_position_in_window]
        top_border, bottom_border = current_borders
        free_zone_top = top_border
        free_zone_bottom = 2.0 - bottom_border
        
        print(f"Step {self.step_count}:")
        print(f"  Voiture Y: {self.car_y:.3f}")
        print(f"  Bord haut: {top_border:.3f}")
        print(f"  Bord bas: {bottom_border:.3f}")
        print(f"  Zone libre: [{free_zone_top:.3f}, {free_zone_bottom:.3f}]")
        print(f"  Largeur zone: {free_zone_bottom - free_zone_top:.3f}")
        print()

def test_environment():
    """Test de l'environnement."""
    env = SimpleRacingEnv()
    
    print("Test de l'environnement de course")
    print("=" * 40)
    
    obs = env.reset()
    env.print_state()
    
    for step in range(5):
        action = np.random.randint(0, 3)  # Action aléatoire
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Action: {action} ({'descendre' if action==0 else 'rester' if action==1 else 'monter'})")
        print(f"Reward: {reward}")
        env.print_state()
        
        if terminated:
            print("Collision !")
            break

if __name__ == "__main__":
    test_environment()
