#!/usr/bin/env python3
"""
Test simple pour vérifier la génération des bords.
"""

import numpy as np

def generate_road_segment():
    """Génère un segment de route avec bords haut et bas de façon aléatoire."""
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

def test_border_generation():
    """Test la génération de bords sur plusieurs exemples."""
    print("Test de génération des bords de route")
    print("=" * 50)
    
    for i in range(10):
        top, bottom = generate_road_segment()
        free_zone_top = top
        free_zone_bottom = 2.0 - bottom
        free_zone_width = free_zone_bottom - free_zone_top
        
        print(f"Step {i+1}:")
        print(f"  Bord haut: {top:.3f}")
        print(f"  Bord bas: {bottom:.3f}")
        print(f"  Zone libre: [{free_zone_top:.3f}, {free_zone_bottom:.3f}]")
        print(f"  Largeur zone libre: {free_zone_width:.3f}")
        print(f"  Zone libre existe: {free_zone_width > 0}")
        print()

if __name__ == "__main__":
    test_border_generation()
