import pygame
import pickle

WIDTH, HEIGHT = 800, 600
BLACK = (10, 10, 30)
PATH_COLOR = (255, 105, 180)  # neon pink

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw Your Path - Neon Glow")
clock = pygame.time.Clock()

path_points = []
drawing = False

def draw_glowing_lines(surface, points, color):
    """Draw neon glow effect for the path."""
    if len(points) > 1:
        for glow in [8, 5, 3]:
            # Adjust alpha for outer glow
            glow_color = (*color[:3], 50) if glow != 3 else (*color[:3], 255)
            temp_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.lines(temp_surf, glow_color, False, points, glow)
            surface.blit(temp_surf, (0, 0))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # save path to file
            with open("custom_path.pkl", "wb") as f:
                pickle.dump(path_points, f)
            pygame.quit()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

    if drawing:
        mx, my = pygame.mouse.get_pos()
        path_points.append((mx, my))

    screen.fill(BLACK)
    draw_glowing_lines(screen, path_points, PATH_COLOR)
    pygame.display.update()
    clock.tick(60)
