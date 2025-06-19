import pygame
import numpy as np
import sounddevice as sd
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import time
import random
import colorsys

# Ustawienia audio
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
audio_buffer = np.zeros(BUFFER_SIZE)


class DisplaySettings:
    def __init__(self):
        self.fullscreen = False
        self.window_size = (1280, 720)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            screen = pygame.display.set_mode((0, 0), pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN)
        else:
            screen = pygame.display.set_mode(self.window_size, pygame.OPENGL | pygame.DOUBLEBUF)
        return screen

def find_audio_device():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if 'blackhole' in dev['name'].lower() or 'loopback' in dev['name'].lower():
            print(f"Using audio device: {dev['name']}")
            return i
    print("using default input device")
    return None


# shapes def
def create_cube():
    vertices = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
               (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)]
    return vertices, edges

def create_tetrahedron():
    vertices = [(0,1,0),(-1,-1,1),(1,-1,1),(0,-1,-1)]
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    return vertices, edges

def create_octahedron():
    vertices = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    edges = [(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),
            (2,4),(2,5),(3,4),(3,5)]
    return vertices, edges

def create_pentagonal_pyramid():
    vertices = [(math.cos(2*math.pi*i/5), math.sin(2*math.pi*i/5), -0.5) for i in range(5)] + [(0,0,1.5)]
    edges = [(i,(i+1)%5) for i in range(5)] + [(i,5) for i in range(5)]
    return vertices, edges

def create_icosahedron():
    phi = (1 + math.sqrt(5)) / 2
    vertices = [
        (-1,phi,0),(1,phi,0),(-1,-phi,0),(1,-phi,0),
        (0,-1,phi),(0,1,phi),(0,-1,-phi),(0,1,-phi),
        (phi,0,-1),(phi,0,1),(-phi,0,-1),(-phi,0,1)
    ]
    edges = [
        (0,1),(0,5),(0,7),(0,10),(0,11),(1,5),(1,7),(1,8),(1,9),
        (2,3),(2,4),(2,6),(2,10),(2,11),(3,4),(3,6),(3,8),(3,9),
        (4,5),(4,9),(5,11),(6,7),(6,10),(7,8),(8,9),(10,11)
    ]
    return vertices, edges

def create_stella():
    tetra_v, _ = create_tetrahedron()
    vertices = tetra_v + [(-x,-y,-z) for (x,y,z) in tetra_v]
    edges = [(i,j) for i in range(4) for j in range(4,8) if i != j-4] + \
            [(i,j) for i in range(4) for j in range(i+1,4)] + \
            [(i,j) for i in range(4,8) for j in range(i+1,8)]
    return vertices, edges

def create_torus(R=1.5, r=0.5, N=20, n=10):
    vertices = []
    for i in range(N):
        for j in range(n):
            theta = 2 * math.pi * i / N
            phi = 2 * math.pi * j / n
            x = (R + r*math.cos(phi)) * math.cos(theta)
            y = (R + r*math.cos(phi)) * math.sin(theta)
            z = r * math.sin(phi)
            vertices.append((x,y,z))
    
    edges = []
    for i in range(N):
        for j in range(n):
            edges.append((i*n + j, i*n + (j+1)%n))
            edges.append((i*n + j, ((i+1)%N)*n + j))
    
    return vertices, edges

def create_spiral(points=100, turns=3, radius=1.5):
    vertices = []
    for i in range(points):
        t = turns * 2 * math.pi * i / points
        x = radius * math.cos(t)
        y = radius * math.sin(t)
        z = -1 + 2 * i / points
        vertices.append((x,y,z))
    
    edges = [(i,i+1) for i in range(points-1)]
    return vertices, edges

# shapes list
shapes = [
    ("cube", *create_cube()),
    ("tetrahedron", *create_tetrahedron()),
    ("octahedron", *create_octahedron()),
    ("pent_pyramid", *create_pentagonal_pyramid()),
    ("icosahedron", *create_icosahedron()),
    ("stella", *create_stella()),
    ("torus", *create_torus()),
    ("spiral", *create_spiral())
]

def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = indata[:,0] if any(indata) else np.zeros(frames)

def get_fft(audio_samples):
    windowed = audio_samples * np.hanning(len(audio_samples))
    fft = np.abs(np.fft.rfft(windowed))
    return fft / np.max(fft) if np.max(fft) > 0 else fft

def get_audio_features(fft):
    bass = np.mean(fft[:10]) if len(fft) > 10 else 0
    mid = np.mean(fft[10:30]) if len(fft) > 30 else 0
    treble = np.mean(fft[30:100]) if len(fft) > 100 else 0
    energy = (bass + mid + treble) / 3
    return bass, mid, treble, energy

def draw_wireframe(vertices, edges, color, line_width=1.0):
    glColor3fv(color)
    glLineWidth(line_width)
    glBegin(GL_LINES)
    for i, j in edges:
        if i < len(vertices) and j < len(vertices):
            glVertex3fv(vertices[i])
            glVertex3fv(vertices[j])
    glEnd()

def interpolate_vertices(v1, v2, t):
    t_smooth = t * t * (3 - 2 * t)  # Ease-in-out
    # Dopasowanie liczby wierzchołków
    len_diff = len(v2) - len(v1)
    if len_diff > 0:
        v1 += [v1[-1]] * len_diff
    elif len_diff < 0:
        v2 += [v2[-1]] * (-len_diff)
    
    return [(x1*(1-t_smooth)+x2*t_smooth, 
             y1*(1-t_smooth)+y2*t_smooth, 
             z1*(1-t_smooth)+z2*t_smooth) 
            for (x1,y1,z1),(x2,y2,z2) in zip(v1, v2)]

def main():
    display_settings = DisplaySettings()
    
    pygame.init()
    screen = pygame.display.set_mode(display_settings.window_size, pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Audio Visualizer (F11 - Fullscreen)")

def find_audio_device():
    device = find_audio_device()
    try:
        stream = sd.InputStream(device=device, channels=1,
                              samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE)
        stream.start()
    except Exception as e:
        print(f"Audio error: {e}")
        stream = None


    clock = pygame.time.Clock()
    current_shape = 0
    next_shape = 1
    last_change = time.time()
    in_transition = False
    transition_start = 0
    rotation = [0, 0, 0]
    
    running = True
    while running:
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            
            # Obsługa pełnoekranowego przełączania
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                screen = display_settings.toggle_fullscreen()
                
            # Analiza audio
            fft = get_fft(audio_buffer)
            bass, mid, treble, energy = get_audio_features(fft)
            
            # Automatyczna zmiana kształtu
            now = time.time()
            if not in_transition and (now - last_change > 5 or energy > 1.0):
                in_transition = True
                transition_start = now
                current_shape = next_shape
                next_shape = random.randint(0, len(shapes)-1)
                while next_shape == current_shape:
                    next_shape = random.randint(0, len(shapes)-1)
                last_change = now
            
            # Renderowanie
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            glPushMatrix()
            
            # Animacja
            rotation[0] += 0.3 + bass
            rotation[1] += 0.5 + mid*0.5
            rotation[2] += 0.2 + treble*0.3
            glRotatef(rotation[0], 1, 0, 0)
            glRotatef(rotation[1], 0, 1, 0)
            glRotatef(rotation[2], 0, 0, 1)
            
            scale = 0.7 + energy * 0.3
            glScalef(scale, scale, scale)
            
            hue = (time.time() * 0.1) % 1.0
            color = colorsys.hsv_to_rgb(hue, 0.9, 0.8 + energy*0.2)
            
            # Rysowanie kształtu
            if in_transition:
                t = (now - transition_start) / 0.5  # Krótkie 0.5s przejście
                if t >= 1.0:
                    in_transition = False
                vertices = interpolate_vertices(
                    shapes[current_shape][1], 
                    shapes[next_shape][1], 
                    t
                )
                edges = shapes[next_shape][2]
            else:
                vertices, edges = shapes[current_shape][1], shapes[current_shape][2]
            
            draw_wireframe(vertices, edges, color, 1.0 + energy*2)
            
            glPopMatrix()
            pygame.display.flip()
            clock.tick(60)

        except KeyboardInterrupt:
            running = False

    if stream:
        stream.stop()
    pygame.quit()

if __name__ == "__main__":
    main()