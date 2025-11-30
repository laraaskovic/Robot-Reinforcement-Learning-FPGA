# draw_path.py
import pygame
import pickle

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw Your Path")
clock = pygame.time.Clock()

path_points = []
drawing = False

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
    if len(path_points) > 1:
        pygame.draw.lines(screen, WHITE, False, path_points, 6)
    pygame.display.update()
    clock.tick(60)
