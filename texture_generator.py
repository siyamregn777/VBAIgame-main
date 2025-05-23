import pygame
import os

# Create textures directory if it doesn't exist
if not os.path.exists('textures'):
    os.makedirs('textures')

# Initialize Pygame
pygame.init()

# Wall texture
wall_size = (64, 64)
wall = pygame.Surface(wall_size)
wall.fill((128, 128, 128))  # Gray base
pygame.draw.rect(wall, (100, 100, 100), (0, 0, 64, 32))
pygame.draw.rect(wall, (140, 140, 140), (32, 32, 64, 64))
pygame.image.save(wall, 'textures/wall.png')

# Floor texture
floor_size = (64, 64)
floor = pygame.Surface(floor_size)
floor.fill((139, 69, 19))  # Brown base
pygame.image.save(floor, 'textures/floor.png')

# Ceiling texture
ceiling_size = (64, 64)
ceiling = pygame.Surface(ceiling_size)
ceiling.fill((200, 200, 200))  # Light gray base
pygame.image.save(ceiling, 'textures/ceiling.png')

print("Textures generated successfully!")
pygame.quit()