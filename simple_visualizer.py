import numpy as np
import pygame
import pyaudio
import colorsys

class AudioVisualizer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Audio Visualizer')
        
        # Audio capture setup
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)
        
        # Color and stability parameters
        self.base_hue = 0.5
        self.noise_threshold = 500  # Adjust to control sensitivity
        
    def generate_dynamic_color(self, intensity):
        hue = (self.base_hue + intensity/255) % 1.0
        saturation = 0.7
        value = 0.7
        r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
        return (r, g, b)
    
    def draw_curved_waveform(self, audio_data):
        # Convert audio data to numpy array
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        
        # Check audio intensity
        max_intensity = np.max(np.abs(audio_data))
        
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw circular background with shadow
        circle_radius = min(self.width, self.height) // 3
        shadow_color = (20, 20, 20)
        pygame.draw.circle(self.screen, shadow_color, 
                           (self.width//2, self.height//2), 
                           circle_radius + 10)
        
        # Dynamic color based on audio intensity
        circle_color = self.generate_dynamic_color(max_intensity)
        pygame.draw.circle(self.screen, circle_color, 
                           (self.width//2, self.height//2), 
                           circle_radius)
        
        # Draw baseline
        baseline_points = [
            (0, self.height//2),
            (self.width, self.height//2)
        ]
        pygame.draw.line(self.screen, (100, 100, 100), 
                         baseline_points[0], baseline_points[1], 2)
        
        # Only draw waveform if sound is above threshold
        if max_intensity > self.noise_threshold:
            # Normalized audio data
            normalized_data = audio_data / np.max(np.abs(audio_data))
            
            # Curved waveform
            points = []
            for i, sample in enumerate(normalized_data):
                x = int(i * (self.width / len(normalized_data)) * 2)
                y = int(self.height/2 + sample * self.height/4)
                points.append((x, y))
            
            # Smooth curved line
            if len(points) > 2:
                pygame.draw.lines(self.screen, (255, 255, 255), False, points, 3)
        
        pygame.display.flip()
    
    def start_visualization(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Capture audio data
            audio_data = self.stream.read(self.CHUNK)
            self.draw_curved_waveform(audio_data)
            
            pygame.time.delay(10)
        
        self.cleanup()
    
    def cleanup(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        pygame.quit()

def main():
    visualizer = AudioVisualizer()
    visualizer.start_visualization()

if __name__ == "__main__":
    main()