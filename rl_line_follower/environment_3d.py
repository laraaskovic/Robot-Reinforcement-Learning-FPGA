# environment_3d.py
import pygame
import numpy as np
import math
import pickle

WIDTH, HEIGHT = 800, 600
BLACK = (10, 10, 30)
ROBOT_COLOR = (255, 105, 180)  # neon pink
SENSOR_COLOR = (255, 50, 50)
TRAIL_COLOR = (100, 200, 255, 50)  # faint cyan trail

class LineFollowEnv3D:
    def __init__(self, path_file="custom_path.pkl"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("RL Line Follower - 3D Neon")
        self.clock = pygame.time.Clock()

        # Robot state
        self.x = 100.0
        self.y = HEIGHT // 2
        self.angle = 0.0
        self.speed = 3.2
        self.sensor_offset = 18
        self.trail = []

        # Load drawn path
        try:
            with open(path_file, "rb") as f:
                self.path_points = pickle.load(f)
        except FileNotFoundError:
            # fallback sinusoidal path
            self.path_points = [(x, int(HEIGHT//2 + 80 * math.sin(x / 150.0))) for x in range(0, WIDTH, 4)]

        self.step_count = 0
        self.max_steps = 1000

    def reset(self):
        self.x = 100.0
        self.y = HEIGHT // 2 + (np.random.rand() - 0.5) * 10
        self.angle = 0.0
        self.step_count = 0
        self.trail = []
        return self._get_observation()

    def _sensor_world_pos(self):
        left_x = int(self.x + math.cos(self.angle + math.pi/4) * self.sensor_offset)
        left_y = int(self.y + math.sin(self.angle + math.pi/4) * self.sensor_offset)
        right_x = int(self.x + math.cos(self.angle - math.pi/4) * self.sensor_offset)
        right_y = int(self.y + math.sin(self.angle - math.pi/4) * self.sensor_offset)
        return (left_x, left_y), (right_x, right_y)

    def _get_sensor_values(self):
        (lx, ly), (rx, ry) = self._sensor_world_pos()
        lx = max(0, min(WIDTH-1, lx))
        ly = max(0, min(HEIGHT-1, ly))
        rx = max(0, min(WIDTH-1, rx))
        ry = max(0, min(HEIGHT-1, ry))
        left_val = 1 if self.screen.get_at((lx, ly))[0] > 200 else 0
        right_val = 1 if self.screen.get_at((rx, ry))[0] > 200 else 0
        return left_val, right_val

    def _closest_path_info(self):
        best_d = float("inf")
        best_idx = 0
        for i, (px, py) in enumerate(self.path_points):
            d = (px - self.x)**2 + (py - self.y)**2
            if d < best_d:
                best_d = d
                best_idx = i
        px, py = self.path_points[best_idx]
        if best_idx + 1 < len(self.path_points):
            nx, ny = self.path_points[best_idx + 1]
        else:
            nx, ny = self.path_points[best_idx - 1]
        tangent = math.atan2(ny - py, nx - px)
        dx = self.x - px
        dy = self.y - py
        offset = dx * math.sin(tangent) - dy * math.cos(tangent)
        return offset, tangent, (px, py)

    def _get_observation(self):
        left, right = self._get_sensor_values()
        offset, tangent, _ = self._closest_path_info()
        norm_offset = np.tanh(offset / 80.0)
        heading_error = math.atan2(math.sin(self.angle - tangent), math.cos(self.angle - tangent))
        norm_heading = np.tanh(heading_error * 2.0)
        norm_speed = np.tanh(self.speed / 10.0)
        return np.array([left, right, norm_offset, norm_heading, norm_speed], dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.angle -= 0.12
        elif action == 1:
            self.angle += 0.12
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        self.trail.append((self.x, self.y))
        if len(self.trail) > 100:
            self.trail.pop(0)

        self.step_count += 1
        left, right = self._get_sensor_values()
        offset, tangent, _ = self._closest_path_info()
        reward = 0.0
        reward -= abs(offset)/20.0
        if abs(offset) < 6:
            reward += 2.5
        heading_error = math.atan2(math.sin(self.angle - tangent), math.cos(self.angle - tangent))
        reward += max(0, 1.5 - abs(heading_error)*2)
        if left or right:
            reward += 0.5

        done = False
        if self.x < 0 or self.x >= WIDTH or self.y < 0 or self.y >= HEIGHT:
            reward -= 25
            done = True
        if abs(offset) > 120 or self.step_count >= 1000:
            done = True

        return self._get_observation(), reward, done

    def render(self, show_sensors=True, show_trail=True, info=None, final_path=False):
        self.screen.fill(BLACK)

        # draw path
        if final_path:
            # smooth neon path with highlights
            if len(self.path_points) > 1:
                pygame.draw.aalines(self.screen, ROBOT_COLOR, False, self.path_points)
                for i in range(0, len(self.path_points), 5):
                    x, y = self.path_points[i]
                    pygame.draw.circle(self.screen, (255, 200, 230), (x, y), 3)
        else:
            # robot training path (optional simple line)
            if len(self.path_points) > 1:
                pygame.draw.lines(self.screen, ROBOT_COLOR, False, self.path_points, 4)

        # faint trail
        if show_trail and len(self.trail) > 1:
            for i, (x, y) in enumerate(self.trail):
                alpha = int(50 * i / len(self.trail))
                trail_surf = pygame.Surface((10, 10), pygame.SRCALPHA)
                pygame.draw.circle(trail_surf, (100, 200, 255, alpha), (5, 5), 5)
                self.screen.blit(trail_surf, (int(x-5), int(y-5)))

        # robot with 3D effect
        pygame.draw.circle(self.screen, ROBOT_COLOR, (int(self.x), int(self.y)), 10)
        pygame.draw.circle(self.screen, (255, 180, 220), (int(self.x-3), int(self.y-3)), 6)

        # sensors
        if show_sensors:
            (lx, ly), (rx, ry) = self._sensor_world_pos()
            pygame.draw.circle(self.screen, SENSOR_COLOR, (lx, ly), 4)
            pygame.draw.circle(self.screen, SENSOR_COLOR, (rx, ry), 4)

        # info overlay
        if info:
            font = pygame.font.SysFont("Arial", 18, bold=True)
            y0 = 5
            for key, val in info.items():
                txt = font.render(f"{key}: {val}", True, (255, 255, 255))
                self.screen.blit(txt, (5, y0))
                y0 += 20

        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
