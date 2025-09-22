import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from collections import deque

class RacingEnv(gym.Env):
    """
    Environnement de course automobile avec scrolling horizontal.
    La voiture évolue verticalement dans une route définie par des bords qui défilent.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None, config=None, **kwargs):
        super().__init__()
        
        # Configuration par défaut
        default_config = {
            'min_road_width': 0.2,           # Largeur minimale de la route
            'window_horizon': 10,             # Fenêtre temporelle (steps)
            'car_position_in_window': 3,      # Position de la voiture dans la fenêtre (0-indexed) - step 3 = présent
            'future_uncertainty': 0.1,        # Niveau d'incertitude futur
            'max_vertical_speed': 0.1,        # Vitesse verticale max
            'max_episode_steps': 1000,        # Durée max d'un épisode
            'scroll_speed': 1.0,              # Vitesse de défilement (1 step par seconde)
            'road_height': 2.0,               # Hauteur totale de la route
            'image_observation': False,       # Observation sous forme d'image
            'border_variation': 0.2           # Variation max des bords (20% de la hauteur)
        }
        
        if config:
            default_config.update(config)
        # Support kwargs passed by gym.make(**env_make_params)
        if kwargs:
            default_config.update(kwargs)
        self.config = default_config
        
        # Espace d'observation
        if self.config['image_observation']:
            # Observation sous forme d'image (fenêtre temporelle)
            self.observation_space = spaces.Box(
                low=0, high=255, 
                shape=(64, self.config['window_horizon'], 3), 
                dtype=np.uint8
            )
        else:
            # Observation vectorielle : [y, vy, 10x(bord_haut, bord_bas)]
            obs_dim = 2 + 2 * self.config['window_horizon']  # y + vy + 10 couples de bords
            self.observation_space = spaces.Box(
                low=np.array([0.0, -self.config['max_vertical_speed']] + 
                           [0.0] * (2 * self.config['window_horizon']), dtype=np.float32),
                high=np.array([self.config['road_height'], self.config['max_vertical_speed']] + 
                            [1.0] * (2 * self.config['window_horizon']), dtype=np.float32)
            )
        
        # Actions : 0=descendre, 1=rester, 2=monter
        self.action_space = spaces.Discrete(3)
        
        # Variables d'état
        self.car_y = 1.0  # Position verticale de la voiture
        self.car_vy = 0.0  # Vitesse verticale de la voiture
        self.step_count = 0
        self.road_history = deque(maxlen=self.config['window_horizon'])
        
        # Rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = (800, 400)
        self.road_width_pixels = 600
        
    def _generate_road_segment(self, step_offset=0):
        """Génère un segment de route avec bords haut et bas de façon aléatoire."""
        # Génération complètement aléatoire des bords
        # Les bords définissent la zone libre : [top_border, 2 - bottom_border]
        
        # Bord haut (0 à 1) - position aléatoire
        top_border = np.random.uniform(0.1, 0.9)
        
        # Bord bas (0 à 1) - position aléatoire  
        bottom_border = np.random.uniform(0.1, 0.9)
        
        # Vérifier que la zone libre existe (top_border < 2 - bottom_border)
        # Si top_border >= 2 - bottom_border, alors pas de zone libre
        if top_border >= 2.0 - bottom_border:
            # Ajuster pour créer une zone libre
            # On garde le plus petit des deux et on ajuste l'autre
            if top_border < bottom_border:
                # Garder top_border, ajuster bottom_border
                bottom_border = 2.0 - top_border - 0.1  # 0.1 de marge
            else:
                # Garder bottom_border, ajuster top_border  
                top_border = 2.0 - bottom_border - 0.1  # 0.1 de marge
        
        # Clamper les valeurs finales
        top_border = np.clip(top_border, 0.1, 0.9)
        bottom_border = np.clip(bottom_border, 0.1, 0.9)
        
        return top_border, bottom_border
    
    def _get_observation(self):
        """Retourne l'observation actuelle."""
        if self.config['image_observation']:
            return self._get_image_observation()
        else:
            return self._get_vector_observation()
    
    def _get_vector_observation(self):
        """Observation vectorielle : [y, vy, 10x(bord_haut, bord_bas)]"""
        obs = [self.car_y, self.car_vy]
        
        # Ajouter l'historique des bords
        for top_border, bottom_border in self.road_history:
            obs.extend([top_border, bottom_border])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_image_observation(self):
        """Observation sous forme d'image de la fenêtre temporelle."""
        img = np.zeros((64, self.config['window_horizon'], 3), dtype=np.uint8)
        
        for i, (top_border, bottom_border) in enumerate(self.road_history):
            # Convertir les bords en pixels
            top_pixel = int(top_border * 32)  # 0-1 -> 0-32
            bottom_pixel = int((2.0 - bottom_border) * 32)  # 0-1 -> 32-64
            
            # Dessiner la route (zone libre en blanc)
            img[top_pixel:bottom_pixel, i, :] = [255, 255, 255]  # Blanc pour la route
            
            # Dessiner les bords (en rouge)
            img[top_pixel, i, :] = [255, 0, 0]  # Bord haut
            img[bottom_pixel-1, i, :] = [255, 0, 0]  # Bord bas
            
            # Marquer la position de la voiture au step actuel
            if i == self.config['car_position_in_window']:
                car_pixel = int(self.car_y * 32)
                if 0 <= car_pixel < 64:
                    img[car_pixel, i, :] = [0, 255, 0]  # Vert pour la voiture
        
        return img
    
    def reset(self, seed=None, options=None):
        """Reset l'environnement à l'état initial."""
        super().reset(seed=seed)
        
        # Réinitialiser l'état
        self.car_y = 1.0  # Position centrale
        self.car_vy = 0.0
        self.step_count = 0
        self.road_history.clear()
        
        # Générer l'historique initial de la route
        for i in range(self.config['window_horizon']):
            step_offset = i - self.config['car_position_in_window']
            top_border, bottom_border = self._generate_road_segment(step_offset)
            self.road_history.append((top_border, bottom_border))
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_observation(), {}
    
    def step(self, action):
        """Exécute une action et retourne le nouvel état."""
        self.step_count += 1
        
        # Appliquer l'action à la vitesse verticale
        if action == 0:  # Descendre
            self.car_vy = max(-self.config['max_vertical_speed'], 
                             self.car_vy - 0.02)
        elif action == 1:  # Rester immobile
            self.car_vy *= 0.9  # Friction douce
        elif action == 2:  # Monter
            self.car_vy = min(self.config['max_vertical_speed'], 
                             self.car_vy + 0.02)
        
        # Mettre à jour la position verticale
        self.car_y += self.car_vy
        self.car_y = np.clip(self.car_y, 0.0, self.config['road_height'])
        
        # Faire défiler la route (supprimer le plus ancien, ajouter un nouveau)
        self.road_history.popleft()
        
        # Générer le nouveau segment de route
        future_step = self.config['car_position_in_window'] + 6
        top_border, bottom_border = self._generate_road_segment(future_step)
        
        # Ajouter de l'incertitude pour les steps futurs
        if future_step > self.config['car_position_in_window']:
            uncertainty = self.config['future_uncertainty'] * (future_step - self.config['car_position_in_window'])
            top_border += np.random.normal(0, uncertainty)
            bottom_border += np.random.normal(0, uncertainty)
            
            # Clamper les valeurs
            top_border = np.clip(top_border, 0.1, 0.9)
            bottom_border = np.clip(bottom_border, 0.1, 0.9)
        
        self.road_history.append((top_border, bottom_border))
        
        # Calculer la récompense
        reward = self._calculate_reward()
        
        # Vérifier les conditions de fin
        terminated = self._is_collision()
        truncated = self.step_count >= self.config['max_episode_steps']
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _calculate_reward(self):
        """Calcule la récompense basée sur la position de la voiture."""
        # Récupérer les bords actuels (step 3 dans la fenêtre = présent)
        current_borders = list(self.road_history)[self.config['car_position_in_window']]
        top_border, bottom_border = current_borders
        
        # Zone libre : [top_border, 2 - bottom_border]
        free_zone_top = top_border
        free_zone_bottom = 2.0 - bottom_border
        
        # Vérifier si la voiture est dans la zone libre
        if free_zone_top <= self.car_y <= free_zone_bottom:
            center = (free_zone_top + free_zone_bottom) / 2
            half_width = (free_zone_bottom - free_zone_top) / 2
            
            if half_width <= 0:
                return 1.0  # cas dégénéré, mais voiture techniquement "dans" la zone

            # Si exactement au centre, récompense maximale 2.0
            if abs(self.car_y - center) < 1e-8:
                return 2.0

            # Échelle linéaire [bord -> centre] => [1.0 -> 2.0]
            distance_to_nearest_border = min(self.car_y - free_zone_top, free_zone_bottom - self.car_y)
            normalized = max(0.0, min(1.0, distance_to_nearest_border / half_width))
            reward = 1.0 + normalized  # 1 à 2
            return float(max(1.0, min(2.0, reward)))
        else:
            # Collision avec les bords
            return -10.0
    
    def _is_collision(self):
        """Vérifie si la voiture est en collision avec les bords."""
        current_borders = list(self.road_history)[self.config['car_position_in_window']]
        top_border, bottom_border = current_borders
        
        free_zone_top = top_border
        free_zone_bottom = 2.0 - bottom_border
        
        return not (free_zone_top <= self.car_y <= free_zone_bottom)
    
    def render(self):
        """Rendu visuel de l'environnement selon le schéma fourni."""
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(self.window_size)
                pygame.display.set_caption("Course Automobile - Scrolling Horizontal")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            
            canvas = pygame.Surface(self.window_size)
            canvas.fill((240, 240, 240))  # Fond gris clair comme le schéma
            
            # Dimensions pour le rendu
            road_start_x = 100
            road_width = 600
            road_height = 300
            road_y_offset = 50
            step_width = road_width // self.config['window_horizon']
            
            # Couleurs selon le schéma
            past_color = (173, 216, 230)      # Bleu clair pour le passé
            present_color = (144, 238, 144)   # Vert clair pour le présent
            future_color = (255, 182, 193)    # Rouge clair pour le futur
            border_color = (139, 69, 19)      # Marron pour les bords
            car_color = (255, 0, 0)           # Rouge pour la voiture
            
            # Dessiner la grille de fond
            for i in range(self.config['window_horizon'] + 1):
                x = road_start_x + i * step_width
                pygame.draw.line(canvas, (200, 200, 200), (x, road_y_offset), (x, road_y_offset + road_height), 1)
            
            for i in range(11):  # 11 lignes horizontales (0.0 à 2.0)
                y = road_y_offset + i * (road_height // 10)
                pygame.draw.line(canvas, (200, 200, 200), (road_start_x, y), (road_start_x + road_width, y), 1)
            
            # Dessiner la route (fenêtre temporelle)
            for i, (top_border, bottom_border) in enumerate(self.road_history):
                x = road_start_x + i * step_width
                
                
                # Couleur de fond selon la zone temporelle
                if i < 3:  # Passé (steps 0-2)
                    bg_color = past_color
                elif i == 3:  # Présent (step 3)
                    bg_color = present_color
                else:  # Futur (steps 4-9)
                    bg_color = future_color
                
                # Dessiner le fond de la zone
                pygame.draw.rect(canvas, bg_color, (x, road_y_offset, step_width, road_height))
                
                # Convertir les bords en pixels (y=0 en bas, y=2 en haut)
                # Zone libre : [top_border, 2 - bottom_border]
                free_zone_top = top_border
                free_zone_bottom = 2.0 - bottom_border
                
                # Convertir en pixels (y=0 en bas, y=2 en haut)
                top_pixel = road_y_offset + int((2.0 - free_zone_top) * road_height / 2)
                bottom_pixel = road_y_offset + int((2.0 - free_zone_bottom) * road_height / 2)
                
                # Dessiner les bords (barres marron verticales)
                # Bord haut (de 0 à top_border) - zone interdite en haut
                pygame.draw.rect(canvas, border_color, 
                               (x, road_y_offset, 8, top_pixel - road_y_offset))
                # Bord bas (de bottom_border à 2) - zone interdite en bas  
                pygame.draw.rect(canvas, border_color, 
                               (x, bottom_pixel, 8, road_y_offset + road_height - bottom_pixel))
                
                # Dessiner la zone libre (blanc) - zone autorisée
                y_start = min(top_pixel, bottom_pixel)
                y_end = max(top_pixel, bottom_pixel)
                if y_end > y_start:  # S'assurer qu'il y a une zone libre
                    pygame.draw.rect(canvas, (255, 255, 255), 
                                   (x + 8, y_start, step_width - 8, y_end - y_start))
            
            # Dessiner la voiture (X rouge au step présent)
            # Le car_position_in_window est 3 (0-indexed), donc step 3 = présent
            car_x = road_start_x + self.config['car_position_in_window'] * step_width + step_width // 2
            car_pixel = road_y_offset + int((2.0 - self.car_y) * road_height / 2)
            
            # Dessiner un X rouge
            cross_size = 8
            pygame.draw.line(canvas, car_color, 
                           (car_x - cross_size, car_pixel - cross_size), 
                           (car_x + cross_size, car_pixel + cross_size), 3)
            pygame.draw.line(canvas, car_color, 
                           (car_x + cross_size, car_pixel - cross_size), 
                           (car_x - cross_size, car_pixel + cross_size), 3)
            
            # Légende
            font = pygame.font.Font(None, 20)
            legend_items = [
                ("Passé", past_color),
                ("Présent", present_color), 
                ("Futur", future_color),
                ("Voiture", car_color)
            ]
            
            legend_x = road_start_x + road_width + 20
            legend_y = road_y_offset + 20
            
            for i, (text, color) in enumerate(legend_items):
                pygame.draw.rect(canvas, color, (legend_x, legend_y + i * 25, 15, 15))
                text_surface = font.render(text, True, (0, 0, 0))
                canvas.blit(text_surface, (legend_x + 20, legend_y + i * 25 + 2))
            
            # Informations textuelles
            info_font = pygame.font.Font(None, 18)
            current_borders = list(self.road_history)[self.config['car_position_in_window']]
            info_text = [
                f"Step: {self.step_count}",
                f"Position Y: {self.car_y:.2f}",
                f"Vitesse Y: {self.car_vy:.3f}",
                f"Zone libre: [{current_borders[0]:.2f}, {2.0 - current_borders[1]:.2f}]"
            ]
            
            for i, text in enumerate(info_text):
                text_surface = info_font.render(text, True, (0, 0, 0))
                canvas.blit(text_surface, (10, 10 + i * 20))
            
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            # Vitesse de défilement : 1 step par seconde
            self.clock.tick(1)  # 1 FPS = 1 step par seconde
    
    def close(self):
        """Ferme l'environnement."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# Test simple
if __name__ == "__main__":
    # Configuration pour le test
    config = {
        'window_horizon': 10,
        'car_position_in_window': 3,
        'future_uncertainty': 0.1,
        'max_vertical_speed': 0.1,
        'max_episode_steps': 200,
        'min_road_width': 0.2
    }
    
    env = RacingEnv(render_mode="human", config=config)
    
    print("Test de l'environnement de course automobile")
    print("Actions: 0=descendre, 1=rester, 2=monter")
    print("Appuyez sur Ctrl+C pour arrêter")
    
    try:
        for episode in range(3):
            obs, _ = env.reset()
            print(f"\nÉpisode {episode + 1}")
            print(f"Observation shape: {obs.shape}")
            
            for step in range(100):
                action = env.action_space.sample()  # Action aléatoire
                obs, reward, terminated, truncated, info = env.step(action)
                
                if step % 20 == 0:  # Afficher moins souvent
                    print(f"  Step {step}: Action={action}, Reward={reward:.2f}")
                
                if terminated or truncated:
                    print(f"  Épisode terminé au step {step}")
                    break
    except KeyboardInterrupt:
        print("\nTest interrompu par l'utilisateur")
    
    env.close()
