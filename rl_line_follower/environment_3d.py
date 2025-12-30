# environment_3d.py
import collections
import math
import pickle
import random

import numpy as np
import pygame

WIDTH, HEIGHT = 800, 600
BLACK = (10, 10, 30)
ROBOT_COLOR = (255, 105, 180)  # neon pink
SENSOR_COLOR = (255, 50, 50)
TRAIL_COLOR = (100, 200, 255, 50)  # faint cyan trail

class LineFollowEnv3D:
    def __init__(
        self,
        path_file="custom_path.pkl",
        partial_obs=False,
        latency_steps=0,
        actuation_limit=0.14,
        speed_limit_range=(2.5, 4.2),
        noise_std=0.05,
        bias_drift=0.0008,
        domain_randomization=True,
        headless=False,
    ):
        pygame.init()
        flags = pygame.HIDDEN if headless else 0
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
        pygame.display.set_caption("RL Line Follower - 3D Neon")
        self.clock = pygame.time.Clock()

        # Robot state
        self.x = 100.0
        self.y = HEIGHT // 2
        self.angle = 0.0
        self.speed = 3.2
        self.sensor_offset = 18
        self.trail = []

        # Domain + hardware realism knobs
        self.partial_obs = partial_obs
        self.domain_randomization = domain_randomization
        self.base_noise_std = noise_std
        self.base_bias_drift = bias_drift
        self.speed_limit_range = speed_limit_range
        self.base_turn_limit = actuation_limit
        self.curvature_range = (10.0, 60.0)  # px amplitude applied to base path
        self.headless = headless

        # Control channel realism
        self.latency_steps = max(0, int(latency_steps))
        self.action_queue = collections.deque(
            [2] * (self.latency_steps + 1), maxlen=self.latency_steps + 1
        )
        self.saturation_events = 0
        self.last_steer = 0.0

        # Load drawn path (kept as a template to randomize against)
        self.base_path_points = self._load_base_path(path_file)
        self.path_points = list(self.base_path_points)
        self.path_surface = pygame.Surface((WIDTH, HEIGHT))

        # Dynamic domain params initialized in reset
        self.speed_limit = np.mean(self.speed_limit_range)
        self.turn_limit = self.base_turn_limit
        self.sensor_noise_std = self.base_noise_std
        self.sensor_bias = 0.0
        self.sensor_bias_drift = self.base_bias_drift
        self.actuation_noise_std = 0.01

        self.step_count = 0
        self.max_steps = 1000
        self._configure_domain()
        self._build_path_surface()

    def _load_base_path(self, path_file):
        try:
            with open(path_file, "rb") as f:
                pts = pickle.load(f)
                return pts
        except FileNotFoundError:
            # fallback sinusoidal path
            return [
                (x, int(HEIGHT // 2 + 80 * math.sin(x / 150.0)))
                for x in range(0, WIDTH, 4)
            ]

    def _generate_fallback_path(self, curvature_gain=40.0, phase=0.0):
        return [
            (
                x,
                int(
                    HEIGHT // 2
                    + curvature_gain * math.sin(x / 140.0 + phase)
                    + random.uniform(-4, 4)
                ),
            )
            for x in range(0, WIDTH, 4)
        ]

    def _randomized_path(self, curvature_gain, phase):
        if not self.base_path_points:
            return self._generate_fallback_path(curvature_gain, phase)
        jittered = []
        for i, (px, py) in enumerate(self.base_path_points):
            wobble = curvature_gain * math.sin(i / 35.0 + phase)
            ny = int(np.clip(py + wobble, 8, HEIGHT - 8))
            jittered.append((int(px), ny))
        return jittered

    def _configure_domain(self):
        if self.domain_randomization:
            curvature = random.uniform(*self.curvature_range)
            phase = random.uniform(-math.pi, math.pi)
            self.path_points = self._randomized_path(curvature, phase)
            self.speed_limit = random.uniform(*self.speed_limit_range)
            self.turn_limit = random.uniform(self.base_turn_limit * 0.6, self.base_turn_limit)
            self.sensor_noise_std = random.uniform(0.6, 1.5) * self.base_noise_std
            self.sensor_bias_drift = random.uniform(0.6, 1.4) * self.base_bias_drift
            self.actuation_noise_std = random.uniform(0.5, 1.4) * 0.02
        else:
            self.path_points = list(self.base_path_points)
            self.speed_limit = np.mean(self.speed_limit_range)
            self.turn_limit = self.base_turn_limit
            self.sensor_noise_std = self.base_noise_std
            self.sensor_bias_drift = self.base_bias_drift
            self.actuation_noise_std = 0.01
        self.domain_params = {
            "speed_limit": round(self.speed_limit, 3),
            "turn_limit": round(self.turn_limit, 4),
            "sensor_noise": round(self.sensor_noise_std, 4),
            "bias_drift": round(self.sensor_bias_drift, 5),
        }
        self._build_path_surface()

    def _build_path_surface(self):
        self.path_surface.fill(BLACK)
        if len(self.path_points) > 1:
            for glow in [6, 4, 2]:
                pygame.draw.lines(self.path_surface, ROBOT_COLOR, False, self.path_points, glow)

    def reset(self):
        self.x = 100.0
        self.y = HEIGHT // 2 + (np.random.rand() - 0.5) * 10
        self.angle = 0.0
        self.speed = min(self.speed_limit, 3.2)
        self.step_count = 0
        self.trail = []
        self.saturation_events = 0
        self.sensor_bias = 0.0
        self.action_queue = collections.deque(
            [2] * (self.latency_steps + 1), maxlen=self.latency_steps + 1
        )
        self._configure_domain()
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
        left_val = 1.0 if self.path_surface.get_at((lx, ly))[0] > 200 else 0.0
        right_val = 1.0 if self.path_surface.get_at((rx, ry))[0] > 200 else 0.0

        # Sensor noise + slow bias drift
        self.sensor_bias += np.random.normal(0.0, self.sensor_bias_drift)
        noise_l = np.random.normal(0.0, self.sensor_noise_std)
        noise_r = np.random.normal(0.0, self.sensor_noise_std)
        left_val = float(np.clip(left_val + noise_l + self.sensor_bias, 0.0, 1.0))
        right_val = float(np.clip(right_val + noise_r + self.sensor_bias, 0.0, 1.0))
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
        if self.partial_obs:
            # Hide geometric state to force memory-based policies
            norm_offset = 0.0
            norm_heading = 0.0
        return np.array([left, right, norm_offset, norm_heading, norm_speed], dtype=np.float32)

    def step(self, action):
        # Simulate control latency
        self.action_queue.append(action)
        effective_action = self.action_queue.popleft()

        steer_cmd = 0.0
        if effective_action == 0:
            steer_cmd = -self.turn_limit
        elif effective_action == 1:
            steer_cmd = self.turn_limit

        steer_cmd += np.random.normal(0.0, self.actuation_noise_std)
        steer_cmd = float(np.clip(steer_cmd, -self.turn_limit, self.turn_limit))
        if abs(steer_cmd) >= self.turn_limit * 0.999:
            self.saturation_events += 1

        self.angle += steer_cmd
        self.last_steer = steer_cmd

        # Speed is bounded by randomized limit
        self.speed = float(np.clip(self.speed + np.random.normal(0.0, 0.05), 0.6 * self.speed_limit, self.speed_limit))
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
        self.screen.blit(self.path_surface, (0, 0))

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

        if not self.headless:
            pygame.display.update()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

    def inject_disturbance(self, angle_kick=0.25, lateral_kick=8.0):
        """Apply a bounded perturbation for recovery testing."""
        self.angle += np.clip(angle_kick, -0.6, 0.6)
        self.x += lateral_kick * math.cos(self.angle + math.pi / 2)
        self.y += lateral_kick * math.sin(self.angle + math.pi / 2)
