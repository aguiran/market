# 🏁 Environnement de Course Automobile

Environnement personnalisé pour l'apprentissage par renforcement : course automobile avec scrolling horizontal.

## 🎯 Concept

- **Voiture** : Point qui évolue verticalement (y ∈ [0, 2])
- **Route** : Définie par des bords haut/bas qui défilent horizontalement
- **Actions** : 0=descendre, 1=rester, 2=monter
- **Objectif** : Maximiser le temps de survie en évitant les bords

## 📊 Observation

### Fenêtre temporelle (10 steps)
- **Passé** (steps -3 à -1) : Historique des bords déjà vus
- **Présent** (step 0) : Position voiture + bords actuels  
- **Futur** (steps +1 à +6) : Prévisions bruitées des bords

### Format vectoriel
```
[y, vy, bord_haut_1, bord_bas_1, ..., bord_haut_10, bord_bas_10]
```

### Format image (optionnel)
```
Image 64x10x3 : Visualisation de la fenêtre temporelle
```

## ⚙️ Configuration

```python
config = {
    'window_horizon': 10,           # Fenêtre temporelle
    'car_position_in_window': 4,    # Position voiture dans fenêtre
    'future_uncertainty': 0.1,      # Incertitude des prévisions
    'max_vertical_speed': 0.1,      # Vitesse verticale max
    'max_episode_steps': 1000,      # Durée max épisode
    'min_road_width': 0.2,          # Largeur minimale route
    'image_observation': False       # Observation image vs vectorielle
}
```

## 🎮 Utilisation

### Test simple
```bash
cd custom_env
python my_custom_env.py
```

### Test avancé
```bash
python test_racing.py
```

### Intégration avec DQN
```python
from custom_env import RacingEnv

env = RacingEnv(render_mode="human", config=config)
obs, _ = env.reset()

for step in range(1000):
    action = env.action_space.sample()  # ou stratégie intelligente
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break
```

## 🏆 Système de récompenses

- **+1.0** : Rester dans la zone libre
- **+0.5** : Bonus pour rester au centre de la route
- **-0.1** : Pénalité pour mouvements excessifs
- **-10.0** : Collision avec les bords

## 🎨 Rendu visuel

- **Fond noir** : Environnement
- **Rectangles blancs** : Zone libre de la route
- **Lignes rouges** : Bords de la route
- **Point vert** : Position de la voiture
- **Informations** : Step, position, vitesse, zone libre

## 📁 Structure

```
custom_env/
├── __init__.py              # Package Python
├── my_custom_env.py         # Environnement RacingEnv
├── test_racing.py          # Script de test avancé
└── README.md               # Ce fichier
```