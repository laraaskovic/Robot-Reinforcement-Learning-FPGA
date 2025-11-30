import numpy as np
import pygame
import math

class LineFollowerEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()

        # Robot parameters
        self.x = 100
        self.y = 300
        self.angle = 0
        self.speed = 2

        # Line path (simple horizontal line)
        self.line_y = 300

    def reset(self):
        self.x = 100
        self.y = 300
        self.angle = 0
        return self._get_sensor_readings()

    def _get_sensor_readings(self):
        # Two sensor positions
        sensor_offset = 20
        left_sensor_x = self.x + math.cos(self.angle) * 10 - math.sin(self.angle) * sensor_offset
        left_sensor_y = self.y + math.sin(self.angle) * 10 + math.cos(self.angle) * sensor_offset

        right_sensor_x = self.x + math.cos(self.angle) * 10 + math.sin(self.angle) * sensor_offset
        right_sensor_y = self.y + math.sin(self.angle) * 10 - math.cos(self.angle) * sensor_offset

        # Sensor detects if it's close to the white line
        left_value = 1 if abs(left_sensor_y - self.line_y) < 5 else 0
        right_value = 1 if abs(right_sensor_y - self.line_y) < 5 else 0

        return np.array([left_value, right_value])

    def step(self, action):
        # Actions: 0 = left, 1 = right, 2 = forward
        if action == 0:
            self.angle -= 0.1
        elif action == 1:
            self.angle += 0.1

        # Move robot
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # Compute reward
        sensors = self._get_sensor_readings()
        reward = 1 if sensors.sum() > 0 else -1

        # Check if robot leaves screen
        done = self.x < 0 or self.x > 600 or self.y < 0 or self.y > 600

        return sensors, reward, done

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.line(self.screen, (255, 255, 255), (0, self.line_y), (600, self.line_y), 5)

        # Draw robot
        pygame.draw.circle(self.screen, (0, 255, 0), (int(self.x), int(self.y)), 8)

        pygame.display.update()
        self.clock.tick(60)
