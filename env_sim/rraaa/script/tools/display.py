#!/usr/bin/env python3

import pygame
import numpy as np
from PIL import Image


class DisplayManager:
    def __init__(self, grid_size, window_size):
        
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return
        for s in self.sensor_list:
            s.render()
        pygame.display.flip()

    def save_screenshot(self, filename="screenshot.png", dpi=600):
        raw = pygame.image.tostring(self.display, "RGB")
        w, h = self.display.get_size()
        img = Image.frombytes("RGB", (w, h), raw)
        img.save(filename, dpi=(dpi, dpi))
        print(f"Saved screenshot: {filename} ({w}x{h}px, {dpi} DPI)")

    def save_row_raw(self, row=0, filename="row_raw.png", dpi=600):
        """Stitch raw sensor data for a row — no upscaling."""
        row_sensors = [s for s in self.sensor_list
                       if hasattr(s, 'display_pos') and s.display_pos[0] == row
                       and hasattr(s, 'data') and s.data is not None]
        row_sensors.sort(key=lambda s: s.display_pos[1])

        arrays = [s.data for s in row_sensors]
        stitched = np.concatenate(arrays, axis=1)
        img = Image.fromarray(stitched.astype(np.uint8))
        img.save(filename, dpi=(dpi, dpi))
        print(f"Saved {filename} — {img.size[0]}x{img.size[1]}px at {dpi} DPI")

    def destroy(self):
        for s in self.sensor_list:
            try:
                s.destroy()
            except:
                print('None type')

    def render_enabled(self):
        return self.display != None