#!/usr/bin/env python3
"""
Script de test pour l'environnement de course automobile.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_env import RacingEnv
import numpy as np

def test_racing_env():
    """Test complet de l'environnement de course."""
    
    # Configuration
    config = {
        'window_horizon': 10,
        'car_position_in_window': 3,  # Step 3 = présent
        'future_uncertainty': 0.1,
        'max_vertical_speed': 0.15,
        'max_episode_steps': 500,
        'min_road_width': 0.3,
        'border_variation': 0.2,  # 20% de variation max
        'image_observation': False  # Test avec observation vectorielle
    }
    
    print("🏁 Test de l'environnement de course automobile")
    print("=" * 50)
    
    # Créer l'environnement
    env = RacingEnv(render_mode="human", config=config)
    
    print(f"📊 Espace d'observation: {env.observation_space}")
    print(f"🎮 Espace d'action: {env.action_space}")
    print(f"⚙️  Configuration: {config}")
    print()
    
    try:
        for episode in range(3):
            print(f"🚗 Épisode {episode + 1}")
            obs, _ = env.reset()
            print(f"   Observation initiale shape: {obs.shape}")
            
            total_reward = 0
            steps = 0
            
            for step in range(200):
                # Stratégie simple : rester au centre quand possible
                action = 1  # Rester immobile par défaut
                
                # Analyser l'observation pour prendre une décision
                if len(obs) >= 22:  # Vérifier qu'on a assez d'éléments
                    car_y = obs[0]
                    car_vy = obs[1]
                    
                    # Récupérer les bords actuels (step 3 dans la fenêtre = présent)
                    current_top = obs[2 + 3 * 2]  # Index 8
                    current_bottom = obs[2 + 3 * 2 + 1]  # Index 9
                    
                    free_zone_top = current_top
                    free_zone_bottom = 2.0 - current_bottom
                    center = (free_zone_top + free_zone_bottom) / 2
                    
                    # Décision basée sur la position par rapport au centre
                    if car_y < center - 0.1:
                        action = 2  # Monter
                    elif car_y > center + 0.1:
                        action = 0  # Descendre
                    else:
                        action = 1  # Rester
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if step % 50 == 0:
                    print(f"   Step {step}: Action={action}, Reward={reward:.2f}, Total={total_reward:.1f}")
                
                if terminated:
                    print(f"   💥 Collision au step {step}!")
                    break
                elif truncated:
                    print(f"   ⏰ Épisode terminé (max steps)")
                    break
            
            print(f"   📈 Résultat: {steps} steps, Reward total: {total_reward:.1f}")
            print()
    
    except KeyboardInterrupt:
        print("\n⏹️  Test interrompu par l'utilisateur")
    
    finally:
        env.close()
        print("✅ Test terminé")

if __name__ == "__main__":
    test_racing_env()
