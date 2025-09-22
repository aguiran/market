#!/usr/bin/env python3
"""
Script pour enregistrer l'environnement personnalisé avec Gymnasium.
"""

import gymnasium as gym
from custom_env import RacingEnv

# Enregistrer l'environnement avec Gymnasium
gym.register(
    id='custom_env/RacingEnv-v0',
    entry_point='custom_env:RacingEnv',
    max_episode_steps=1000,
)

if __name__ == "__main__":
    print("Environnement RacingEnv enregistré avec succès !")
    print("ID: custom_env/RacingEnv-v0")
    
    # Test de l'environnement enregistré
    env = gym.make('custom_env/RacingEnv-v0')
    print(f"Environnement créé: {env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    env.close()
