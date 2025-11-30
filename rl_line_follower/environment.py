# environment.py

'''
import pygame
import numpy as np
import math
import random

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ROBOT_COLOR = (0, 200, 255)

class LineFollowEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("RL Line Follower")
        self.clock = pygame.time.Clock()

        # robot state
        self.x = 100.0
        self.y = HEIGHT // 2
        self.angle = 0.0
        self.speed = 0.0

        # geometry
        self.sensor_offset = 18
        self.wheel_speed = 3.2

        # create multiple smooth loopy paths
        self.paths = [self._generate_smooth_path(loop_count=i+1) for i in range(3)]
        self.current_path = 0
        self.path_points = self.paths[self.current_path]

        # draw static path onto surface for fast pixel tests
        self.path_surface = pygame.Surface((WIDTH, HEIGHT))
        self._draw_path()

        # episode bookkeeping
        self.step_count = 0
        self.max_steps = 1000
        self.done = False

    def _generate_smooth_path(self, loop_count=1):
        """Generate a smooth sinusoidal path with a specified number of loops."""
        points = []
        amplitude = 80  # vertical amplitude of loops
        frequency = loop_count * 2 * math.pi / WIDTH  # number of full sine loops over width
        for x in range(0, WIDTH, 4):
            y = HEIGHT//2 + amplitude * math.sin(frequency * x)
            points.append((x, int(y)))
        return points

    def _draw_path(self):
        self.path_surface.fill(BLACK)
        pygame.draw.lines(self.path_surface, WHITE, False, self.path_points, 20)

    def reset(self):
        self.x = 100.0
        self.y = HEIGHT // 2
        self.angle = 0.0
        self.speed = self.wheel_speed
        self.step_count = 0
        self.done = False

        # choose a random path
        self.current_path = random.choice(range(len(self.paths)))
        self.path_points = self.paths[self.current_path]
        self._draw_path()

        return self._get_observation()

    # ----------------- all sensor/observation/step methods remain the same -----------------
    def _sensor_world_pos(self):
        left_x = int(self.x + math.cos(self.angle + math.pi/4) * self.sensor_offset)
        left_y = int(self.y + math.sin(self.angle + math.pi/4) * self.sensor_offset)
        right_x = int(self.x + math.cos(self.angle - math.pi/4) * self.sensor_offset)
        right_y = int(self.y + math.sin(self.angle - math.pi/4) * self.sensor_offset)
        left_x = max(0, min(WIDTH-1, left_x))
        left_y = max(0, min(HEIGHT-1, left_y))
        right_x = max(0, min(WIDTH-1, right_x))
        right_y = max(0, min(HEIGHT-1, right_y))
        return (left_x, left_y), (right_x, right_y)

    def _get_sensor_values(self):
        (lx, ly), (rx, ry) = self._sensor_world_pos()
        left_val = 1 if self.path_surface.get_at((lx, ly))[0] > 200 else 0
        right_val = 1 if self.path_surface.get_at((rx, ry))[0] > 200 else 0
        return left_val, right_val

    def _closest_path_info(self):
        best_d = 1e9
        best_idx = 0
        for i, (px, py) in enumerate(self.path_points):
            d = (px - self.x) ** 2 + (py - self.y) ** 2
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
        self.x += math.cos(self.angle) * self.wheel_speed
        self.y += math.sin(self.angle) * self.wheel_speed

        self.step_count += 1
        left, right = self._get_sensor_values()
        offset, tangent, nearest = self._closest_path_info()
        heading_error = math.atan2(math.sin(self.angle - tangent), math.cos(self.angle - tangent))

        reward = 0.0
        reward -= (abs(offset) / 20.0)
        if abs(offset) < 6:
            reward += 2.5
        reward += max(0.0, 1.5 - abs(heading_error) * 2.0)
        if left or right:
            reward += 0.5

        done = False
        if self.x < 0 or self.x >= WIDTH or self.y < 0 or self.y >= HEIGHT:
            reward -= 25.0
            done = True
        if abs(offset) > 120:
            reward -= 40.0
            done = True
        if self.step_count >= self.max_steps:
            done = True
        if abs(self.angle) > 6.0:
            reward -= 10.0
            done = True

        obs = self._get_observation()
        return obs, reward, done

    def render(self, show_sensors=True):
        self.screen.blit(self.path_surface, (0,0))
        pygame.draw.circle(self.screen, ROBOT_COLOR, (int(self.x), int(self.y)), 10)
        if show_sensors:
            (lx, ly), (rx, ry) = self._sensor_world_pos()
            pygame.draw.circle(self.screen, (255,0,0), (lx, ly), 4)
            pygame.draw.circle(self.screen, (255,0,0), (rx, ry), 4)
            offset, tangent, (px, py) = self._closest_path_info()
            pygame.draw.circle(self.screen, (0,255,0), (int(px), int(py)), 4)
        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
'''


'''
import pygame
import numpy as np
import math

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ROBOT_COLOR = (0, 200, 255)

class LineFollowEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("RL Line Follower")
        self.clock = pygame.time.Clock()

        self.sensor_offset = 18
        self.wheel_speed = 3.2

        # robot state
        self.x = 100.0
        self.y = HEIGHT // 2
        self.angle = 0.0
        self.speed = self.wheel_speed

        # path points
        self.paths = []
        self._create_paths()
        self.current_path_idx = 0
        self.path_points = self.paths[self.current_path_idx]

        # draw static path
        self.path_surface = pygame.Surface((WIDTH, HEIGHT))
        self._draw_path()

        self.step_count = 0
        self.max_steps = 1000
        self.done = False

    def _create_paths(self):
        """Define multiple crazy loopty paths."""
        # path 1: figure-8
        points = []
        for t in range(0, 800, 4):
            x = t
            y = int(300 + 80 * math.sin(t/80.0) - 80 * math.sin(t/160.0))
            points.append((x, y))
        self.paths.append(points)

        # path 2: spiral-like curve
        points = []
        for t in range(0, 800, 4):
            x = t
            y = int(300 + 60 * math.sin(t/50.0) + 40 * math.cos(t/30.0))
            points.append((x, y))
        self.paths.append(points)

        # path 3: S-curve
        points = []
        for t in range(0, 800, 4):
            x = t
            y = int(300 + 100 * math.sin(t / 120.0))
            points.append((x, y))
        self.paths.append(points)

    def _draw_path(self):
        self.path_surface.fill(BLACK)
        pygame.draw.lines(self.path_surface, WHITE, False, self.path_points, 20)

    def reset(self):
        self.x = 100.0
        self.y = HEIGHT // 2
        self.angle = 0.0
        self.speed = self.wheel_speed
        self.step_count = 0
        self.done = False

        # pick a random path
        self.current_path_idx = np.random.choice(len(self.paths))
        self.path_points = self.paths[self.current_path_idx]
        self._draw_path()
        return self._get_observation()

    # ----------------- sensor & observation -----------------
    def _sensor_world_pos(self):
        left_x = int(self.x + math.cos(self.angle + math.pi/4) * self.sensor_offset)
        left_y = int(self.y + math.sin(self.angle + math.pi/4) * self.sensor_offset)
        right_x = int(self.x + math.cos(self.angle - math.pi/4) * self.sensor_offset)
        right_y = int(self.y + math.sin(self.angle - math.pi/4) * self.sensor_offset)
        left_x = max(0, min(WIDTH-1, left_x))
        left_y = max(0, min(HEIGHT-1, left_y))
        right_x = max(0, min(WIDTH-1, right_x))
        right_y = max(0, min(HEIGHT-1, right_y))
        return (left_x, left_y), (right_x, right_y)

    def _get_sensor_values(self):
        (lx, ly), (rx, ry) = self._sensor_world_pos()
        left_val = 1 if self.path_surface.get_at((lx, ly))[0] > 200 else 0
        right_val = 1 if self.path_surface.get_at((rx, ry))[0] > 200 else 0
        return left_val, right_val

    def _closest_path_info(self):
        best_d = 1e9
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
        if action == 0: self.angle -= 0.12
        elif action == 1: self.angle += 0.12
        self.x += math.cos(self.angle) * self.wheel_speed
        self.y += math.sin(self.angle) * self.wheel_speed

        self.step_count += 1
        left, right = self._get_sensor_values()
        offset, tangent, nearest = self._closest_path_info()
        heading_error = math.atan2(math.sin(self.angle - tangent), math.cos(self.angle - tangent))

        reward = 0.0
        reward -= (abs(offset) / 20.0)
        if abs(offset) < 6: reward += 2.5
        reward += max(0.0, 1.5 - abs(heading_error) * 2.0)
        if left or right: reward += 0.5

        done = False
        if self.x < 0 or self.x >= WIDTH or self.y < 0 or self.y >= HEIGHT:
            reward -= 25.0
            done = True
        if abs(offset) > 120:
            reward -= 40.0
            done = True
        if self.step_count >= self.max_steps: done = True
        if abs(self.angle) > 6.0:
            reward -= 10.0
            done = True

        obs = self._get_observation()
        return obs, reward, done

    def render(self, show_sensors=True):
        self.screen.blit(self.path_surface, (0,0))
        pygame.draw.circle(self.screen, ROBOT_COLOR, (int(self.x), int(self.y)), 10)
        if show_sensors:
            (lx, ly), (rx, ry) = self._sensor_world_pos()
            pygame.draw.circle(self.screen, (255,0,0), (lx, ly), 4)
            pygame.draw.circle(self.screen, (255,0,0), (rx, ry), 4)
            offset, tangent, (px, py) = self._closest_path_info()
            pygame.draw.circle(self.screen, (0,255,0), (int(px), int(py)), 4)
        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
'''




# environment.py
import pygame
import numpy as np
import math
import pickle

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ROBOT_COLOR = (0, 200, 255)

class LineFollowEnv:
    def __init__(self, path_file="custom_path.pkl"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("RL Line Follower")
        self.clock = pygame.time.Clock()

        # robot state
        self.x = 100.0
        self.y = HEIGHT // 2
        self.angle = 0.0
        self.speed = 3.2
        self.sensor_offset = 18

        # load custom path
        try:
            with open(path_file, "rb") as f:
                self.path_points = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"{path_file} not found. Run draw_path.py first!")

        # draw static path
        self.path = pygame.Surface((WIDTH, HEIGHT))
        self.path.fill(BLACK)
        if len(self.path_points) > 1:
            pygame.draw.lines(self.path, WHITE, False, self.path_points, 6)

        # episode bookkeeping
        self.step_count = 0
        self.max_steps = 1000

    def reset(self):
        self.x = 100.0
        self.y = HEIGHT // 2 + (np.random.rand() - 0.5) * 10.0
        self.angle = 0.0
        self.step_count = 0
        return self._get_observation()

    def _sensor_world_pos(self):
        # sensors angled forward-left and forward-right
        left_x = int(self.x + math.cos(self.angle + math.pi/4) * self.sensor_offset)
        left_y = int(self.y + math.sin(self.angle + math.pi/4) * self.sensor_offset)
        right_x = int(self.x + math.cos(self.angle - math.pi/4) * self.sensor_offset)
        right_y = int(self.y + math.sin(self.angle - math.pi/4) * self.sensor_offset)
        left_x = max(0, min(WIDTH-1, left_x))
        left_y = max(0, min(HEIGHT-1, left_y))
        right_x = max(0, min(WIDTH-1, right_x))
        right_y = max(0, min(HEIGHT-1, right_y))
        return (left_x, left_y), (right_x, right_y)

    def _get_sensor_values(self):
        (lx, ly), (rx, ry) = self._sensor_world_pos()
        left_val = 1 if self.path.get_at((lx, ly))[0] > 200 else 0
        right_val = 1 if self.path.get_at((rx, ry))[0] > 200 else 0
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

        self.step_count += 1
        offset, tangent, _ = self._closest_path_info()
        left, right = self._get_sensor_values()
        heading_error = math.atan2(math.sin(self.angle - tangent), math.cos(self.angle - tangent))

        reward = 0.0
        reward -= abs(offset) / 20.0
        if abs(offset) < 6:
            reward += 2.5
        reward += max(0.0, 1.5 - abs(heading_error) * 2.0)
        if left or right:
            reward += 0.5

        done = False
        if self.x < 0 or self.x >= WIDTH or self.y < 0 or self.y >= HEIGHT:
            reward -= 25.0
            done = True
        if abs(offset) > 120:
            reward -= 40.0
            done = True
        if self.step_count >= 1000:
            done = True

        return self._get_observation(), reward, done

    def render(self):
        self.screen.blit(self.path, (0,0))
        pygame.draw.circle(self.screen, ROBOT_COLOR, (int(self.x), int(self.y)), 10)
        (lx, ly), (rx, ry) = self._sensor_world_pos()
        pygame.draw.circle(self.screen, (255,0,0), (lx, ly), 4)
        pygame.draw.circle(self.screen, (255,0,0), (rx, ry), 4)
        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

