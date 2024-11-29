import tkinter as tk
from tkinter import ttk
import numpy as np
import threading
import queue
import pyaudio
import struct
import time

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Matplotlib not found. Using basic visualization...")
    MATPLOTLIB_AVAILABLE = False

class VoiceVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Voice Visualizer")
        
        # Configure main window
        self.master.geometry("800x600")
        self.master.configure(bg='black')
        
        # Create frames
        self.viz_frame = ttk.Frame(self.master)
        self.viz_frame.pack(pady=10, expand=True, fill='both')
        
        self.text_frame = ttk.Frame(self.master)
        self.text_frame.pack(pady=10, expand=True, fill='both')
        
        # Audio setup
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        
        # Data queue for thread communication
        self.queue = queue.Queue()
        
        # Setup visualization
        self._setup_visualization()
        
        # Start audio processing
        self._start_audio_processing()
    
    def _setup_visualization(self):
        if MATPLOTLIB_AVAILABLE:
            # Setup matplotlib figure
            self.figure, self.ax = plt.subplots(figsize=(8, 3))
            self.canvas = FigureCanvasTkAgg(self.figure, self.viz_frame)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack(fill='both', expand=True)
            
            # Configure plot
            self.ax.set_facecolor('black')
            self.figure.patch.set_facecolor('black')
            self.ax.set_ylim(-32768, 32768)  # Full range of 16-bit audio
            self.ax.set_xlim(0, self.CHUNK)
            self.line, = self.ax.plot([], [], color='cyan', linewidth=2)
            
            # Remove axes for cleaner look
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        else:
            # Fallback visualization using tkinter Canvas
            self.canvas = tk.Canvas(self.viz_frame, bg='black', height=200)
            self.canvas.pack(fill='both', expand=True)
            self.line = self.canvas.create_line(0, 100, 800, 100, fill='cyan')
        
        # Text display
        self.text_display = tk.Text(self.text_frame, height=10, bg='black', 
                                  fg='cyan', font=('Arial', 12))
        self.text_display.pack(pady=10, padx=10, fill='both', expand=True)
    
    def _start_audio_processing(self):
        try:
            # PyAudio initialization
            self.p = pyaudio.PyAudio()
            
            # Start audio stream
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            # Flag to control thread
            self.running = True
            
            # Start audio capture thread
            self.thread = threading.Thread(target=self._capture_audio, daemon=True)
            self.thread.start()
            
            # Start visualization update
            self.animate()
            
        except Exception as e:
            print(f"Error initializing audio: {e}")
            self.running = False
    
    def _capture_audio(self):
        while self.running:
            try:
                # Read audio data
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                
                # Convert binary data to integer array
                data_int = struct.unpack(f'{self.CHUNK}h', data)
                
                # Put data in queue, don't block if queue is full
                try:
                    self.queue.put(data_int, block=False)
                except queue.Full:
                    # If queue is full, discard oldest data
                    try:
                        self.queue.get_nowait()
                        self.queue.put(data_int, block=False)
                    except queue.Empty:
                        pass
            except Exception as e:
                print(f"Audio capture error: {e}")
                time.sleep(0.1)
    
    def animate(self):
        try:
            if not self.queue.empty():
                try:
                    data = self.queue.get_nowait()
                    
                    if MATPLOTLIB_AVAILABLE:
                        # Matplotlib visualization
                        self.line.set_data(range(len(data)), data)
                        self.ax.set_ylim(min(data), max(data))
                        self.canvas.draw()
                    else:
                        # Simple tkinter visualization
                        points = []
                        for i, val in enumerate(data[::16]):  # Sample every 16th point
                            x = i * 10
                            y = 100 + (val / 128)  # Adjust scaling
                            points.extend([x, y])
                        self.canvas.coords(self.line, *points)
                
                except queue.Empty:
                    pass
            
            # Schedule next update
            self.master.after(30, self.animate)
        
        except Exception as e:
            print(f"Animation error: {e}")
    
    def update_text(self, text, clear=False):
        if clear:
            self.text_display.delete(1.0, tk.END)
        self.text_display.insert(tk.END, text + '\n')
        self.text_display.see(tk.END)
    
    def cleanup(self):
        # Stop audio processing
        self.running = False
        
        # Close audio stream
        if hasattr(self, 'stream'):
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
        
        # Terminate PyAudio
        if hasattr(self, 'p'):
            try:
                self.p.terminate()
            except Exception:
                pass
        
        # Stop Tkinter main loop
        if self.master:
            self.master.quit()

def main():
    root = tk.Tk()
    viz = VoiceVisualizer(root)
    
    def test_update():
        viz.update_text("Testing the visualizer...")
        root.after(2000, lambda: viz.update_text("Another test message"))
    
    root.after(1000, test_update)
    
    # Proper cleanup on window close
    def on_closing():
        viz.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()