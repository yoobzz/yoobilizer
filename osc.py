#v4.131

import pygame
import numpy as np
import sounddevice as sd
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import time
import sys
import colorsys
import random

#audio
SAMPLE_RATE = 44100 
BUFFER_SIZE = 2048   #
audio_buffer = np.zeros(BUFFER_SIZE)

# Parametry analizy audio
BASS_RANGE = (20, 250)    
MID_RANGE = (250, 4000)   
TREBLE_RANGE = (4000, 16000)  


scale_factors = {
    "cube": 5.0,
    "tetrahedron": 6.0,
    "pent_pyramid": 2.0,
    "octahedron": 5.0,
    "icosahedron": 4.0,
    "stella": 4.0,
    "superquadric": 5.0,
    "hyperquadric": 5.0,
    "eye": 4.0,
    "torus": 5.0,
    "spiral": 10.0,
    "vortex": 8.0  # new shape in this v 
}

SPECIAL_CHARS = list("_!+||?_!@^^**__-+++-=<>/\\[]{}()~#$%&*^°±§")
CHAR_GROUPS = [
    r"+++---~~~",
    r"|||///\\\\\\",
    r"===!!!???",
    r"[[[]]]{{{}}}",
    r"^^^***###",
    r"@@@$$$%%%",
    r"&<>/\\~°±§"
]


cube_vertices = [
    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
]
cube_edges = [
    (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
    (4,5),(4,6),(5,7),(6,7)
]

oct_vertices = [
    (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)
]
oct_edges = [
    (0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,4),(2,5),(3,4),(3,5)
]

def pentagonal_pyramid_vertices():
    verts = []
    for i in range(5):
        angle = 2*math.pi*i/5
        verts.append((math.cos(angle), math.sin(angle), -0.5))
    verts.append((0,0,1.5))
    return verts
pent_edges = [(0,1),(1,2),(2,3),(3,4),(4,0)] + [(i,5) for i in range(5)]

tetra_vertices = [
    (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)
]
tetra_edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

phi = (1 + math.sqrt(5)) / 2
ico_vertices = [
    (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
    (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
    (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
]
ico_edges = [
    (0,1),(0,5),(0,7),(0,10),(0,11),(1,5),(1,7),(1,8),(1,9),
    (2,3),(2,4),(2,6),(2,10),(2,11),(3,4),(3,6),(3,8),(3,9),
    (4,5),(4,9),(5,11),(6,7),(6,10),(7,8),(8,9),(10,11)
]

stella_vertices = [(x*1.5,y*1.5,z*1.5) for (x,y,z) in tetra_vertices] + [(-x*1.5,-y*1.5,-z*1.5) for (x,y,z) in tetra_vertices]
stella_edges = [(i,j) for i in range(4) for j in range(4,8) if i != j-4] + [(i,j) for i in range(4) for j in range(i+1,4)] + [(i,j) for i in range(4,8) for j in range(i+1,8)]

def superquadric_vertices(n=40):
    verts = []
    for i in range(n):
        theta = 2 * math.pi * i / n
        for j in range(n):
            phi = math.pi * j / (n-1)
            x = math.cos(theta)**3 * math.sin(phi)**3
            y = math.sin(theta)**3 * math.sin(phi)**3
            z = math.cos(phi)**3
            verts.append((x*3.5, y*3.5, z*3.5))
    return verts

def superquadric_edges(n=40):
    edges = []
    total = n*n
    for i in range(n):
        for j in range(n-1):
            idx = i*n + j
            edges.append((idx, idx+1))
        for j in range(n):
            idx = i + j*n
            if idx + n < total:
                edges.append((idx, idx+n))
    return edges

def hyperquadric_vertices(n=40):
    verts = []
    for i in range(n):
        theta = 2 * math.pi * i / n
        for j in range(n):
            phi = math.pi * j / (n-1)
            x = math.cos(theta)**2 * math.sin(phi)**3
            y = math.sin(theta)**2 * math.sin(phi)**3
            z = math.cos(phi)**2
            verts.append((x*3.5, y*3.5, z*3.5))
    return verts

def hyperquadric_edges(n=40):
    edges = []
    total = n*n
    for i in range(n):
        for j in range(n-1):
            idx = i*n + j
            edges.append((idx, idx+1))
        for j in range(n):
            idx = i + j*n
            if idx + n < total:
                edges.append((idx, idx+n))
    return edges

def eye_vertices(n=30):
    verts = []

    for i in range(n):
        theta = 2 * math.pi * i / n
        for j in range(n//2):
            phi = math.pi * j / (n//2-1)
            x = math.cos(theta) * math.sin(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(phi)
            verts.append((x*3.0, y*3.0, z*3.0))
    

    iris_n = n//2
    iris_start = len(verts)
    for i in range(iris_n):
        theta = 2 * math.pi * i / iris_n
        for j in range(iris_n//2):
            r = 0.3 + 0.7 * j / (iris_n//2-1)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            z = 2.8  # Wysunięta do przodu
            verts.append((x*3.0, y*3.0, z*3.0))
    

    pupil_center = len(verts)
    verts.append((0, 0, 2.9 * 3.0))
    
    return verts

def eye_edges(n=30):
    edges = []
    # Połączenia dla białej części
    for i in range(n):
        for j in range(n//2-1):
            idx = i*(n//2) + j
            edges.append((idx, idx+1))
        for j in range(n//2):
            idx = i + j*n
            if idx + n < n*(n//2):
                edges.append((idx, idx+n))
    

    iris_start = n*(n//2)
    iris_n = n//2
    for i in range(iris_n):
        for j in range(iris_n//2-1):
            idx = iris_start + i*(iris_n//2) + j
            edges.append((idx, idx+1))
        for j in range(iris_n//2):
            idx = iris_start + i + j*iris_n
            if idx + iris_n < iris_start + iris_n*(iris_n//2):
                edges.append((idx, idx+iris_n))
    

    pupil_center = iris_start + iris_n*(iris_n//2)
    for i in range(iris_n):
        edges.append((pupil_center, iris_start + i*(iris_n//2)))
    
    return edges

def spiral_vertices(n=500, turns=80, radius=15.0, length=75):
    verts = []
    for i in range(n):
        t = i / (n-1)
        angle = 2 * math.pi * turns * t
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = length * (t - 0.5)
        verts.append((x, y, z))
    return verts

def spiral_edges(n=500):
    return [(i, i+1) for i in range(n-1)]

def torus_vertices(R=5, r=1.5, N=50, n=20):
    verts = []
    for i in range(N):
        theta = 2 * math.pi * i / N
        for j in range(n):
            phi = 2 * math.pi * j / n
            x = (R + r * math.cos(phi)) * math.cos(theta)
            y = (R + r * math.cos(phi)) * math.sin(theta)
            z = r * math.sin(phi)
            verts.append((x, y, z))
    return verts

def torus_edges(N=50, n=20):
    edges = []
    for i in range(N):
        for j in range(n):
            idx = i * n + j
            edges.append((idx, idx + 1 if j < n-1 else idx - n + 1))
            edges.append((idx, (idx + n) % (N * n)))
    return edges

def vortex_vertices(n=300, turns=50, radius=10.0, length=60):
    verts = []
    for i in range(n):
        t = i / (n-1)
        angle = 2 * math.pi * turns * t
        r = radius * (1 - t*0.8) 
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        z = length * (t - 0.5)
        verts.append((x, y, z))
    return verts

def vortex_edges(n=300):
    edges = [(i, i+1) for i in range(n-1)]

    for i in range(0, n-1, 5):
        edges.append((i, min(i+5, n-1)))
    return edges


spiral_verts = spiral_vertices()
spiral_edges = spiral_edges()
torus_verts = torus_vertices()
torus_edges = torus_edges()
vortex_verts = vortex_vertices()
vortex_edges = vortex_edges()


shapes = [
    ("cube", cube_vertices, cube_edges),
    ("tetrahedron", tetra_vertices, tetra_edges),
    ("pent_pyramid", pentagonal_pyramid_vertices(), pent_edges),
    ("octahedron", oct_vertices, oct_edges),
    ("icosahedron", ico_vertices, ico_edges),
    ("stella", stella_vertices, stella_edges),
    ("superquadric", superquadric_vertices(), superquadric_edges()),
    ("hyperquadric", hyperquadric_vertices(), hyperquadric_edges()),
    ("eye", eye_vertices(), eye_edges()),
    ("torus", torus_verts, torus_edges),
    ("spiral", spiral_verts, spiral_edges),
    ("vortex", vortex_verts, vortex_edges)
]

class AudioAnalyzer:
    def __init__(self, sample_rate, buffer_size):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.fft_freqs = np.fft.rfftfreq(buffer_size, 1.0/sample_rate)
        self.prev_bass = 0
        self.prev_mid = 0
        self.prev_treble = 0
        
    def get_frequency_energy(self, fft, freq_range):
        low, high = freq_range
        mask = (self.fft_freqs >= low) & (self.fft_freqs <= high)
        return np.mean(fft[mask]) if np.any(mask) else 0.0
        
    def analyze(self, audio_samples):
        windowed = audio_samples * np.hanning(len(audio_samples))
        fft = np.abs(np.fft.rfft(windowed))
        if np.max(fft) > 0:
            fft = fft / np.max(fft)
            

        bass = self.get_frequency_energy(fft, BASS_RANGE) ** 1.5
        mid = self.get_frequency_energy(fft, MID_RANGE) ** 1.2
        treble = self.get_frequency_energy(fft, TREBLE_RANGE) ** 0.8
        

        bass = 0.7 * bass + 0.3 * self.prev_bass
        mid = 0.7 * mid + 0.3 * self.prev_mid
        treble = 0.7 * treble + 0.3 * self.prev_treble
        
        self.prev_bass = bass
        self.prev_mid = mid
        self.prev_treble = treble
        
        energy = (bass * 0.4 + mid * 0.3 + treble * 0.3)
        
        return {
            'fft': fft,
            'bass': bass,
            'mid': mid,
            'treble': treble,
            'energy': energy
        }

def audio_callback(indata, frames, time, status):
    global audio_buffer
    if any(indata):
        audio_buffer = np.roll(audio_buffer, -frames)
        audio_buffer[-frames:] = indata[:, 0]

def find_device(name_part):
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if name_part.lower() in device['name'].lower() and device['max_input_channels'] > 0:
            return i
    return None

def neon_color(intensity, base_hue, brightness=1.0):
    saturation = 0.9 + 0.1 * intensity
    value = 0.7 + 0.3 * intensity * brightness
    r, g, b = colorsys.hsv_to_rgb(base_hue, saturation, value)
    return r, g, b

def init_font():
    pygame.font.init()
    return pygame.font.SysFont('Arial', 32)

def draw_floating_chars(font, angle, energy, bass, mid, treble):
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    

    num_groups = int(5 + energy * 5)
    
    for _ in range(num_groups):

        x = random.uniform(-20, 20)
        y = random.uniform(-20, 20)
        z = random.uniform(-20, 20)
        

        char_group = random.choice(CHAR_GROUPS)
        num_chars = random.randint(2, min(5, len(char_group)))
        chars = char_group[:num_chars]
        

        size = int(10 + bass * 30 + treble * 20)
        hue = (angle * 0.01 + mid * 0.5) % 100
        color = (1.0, 1.0, 1.0)  # RGB values for white
        alpha = min(0.3 + energy * 0.7, 1.0)
        

        text_surface = font.render(chars, True, (int(color[0]*255), int(color[1]*255), int(color[2]*255)))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        

        text_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, text_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_surface.get_width(), text_surface.get_height(), 
                    0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        

        glPushMatrix()
        glTranslatef(x, y, z)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, text_id)
        glColor4f(1, 1, 1, alpha)
        

        scale = size / 56.0  # 32 to rozmiar czcionki
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(-scale, -scale, 0)
        glTexCoord2f(1, 0); glVertex3f(scale, -scale, 0)
        glTexCoord2f(1, 1); glVertex3f(scale, scale, 0)
        glTexCoord2f(0, 1); glVertex3f(-scale, scale, 0)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()
        glDeleteTextures([text_id])
    
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_BLEND)

def draw_torus(R, r, N, n, angle, bass, mid, treble):
    glLineWidth(1.0 + treble*3)
    glBegin(GL_LINES)
    
    for i in range(N):
        theta1 = (2 * math.pi * i) / N + angle * 0.02
        theta2 = (2 * math.pi * (i+1)) / N + angle * 0.02
        for j in range(n):
            phi1 = (2 * math.pi * j) / n + angle * 0.05
            phi2 = (2 * math.pi * (j+1)) / n + angle * 0.05
            
            mod_r = r * (1 + 0.2 * bass * math.sin(5 * theta1 + angle) + 
                         0.2 * mid * math.sin(7 * phi1 + angle*2) + 
                         0.1 * treble * math.sin(13 * (theta1+phi1) + angle*3))
            
            points = []
            for theta, phi in [(theta1, phi1), (theta1, phi2), 
                              (theta2, phi1), (theta2, phi2)]:
                x = (R + mod_r * math.cos(phi)) * math.cos(theta)
                y = (R + mod_r * math.cos(phi)) * math.sin(theta)
                z = mod_r * math.sin(phi)
                points.append((x,y,z))
            
            intensity = 0.3 + 0.7 * (bass * 0.4 + mid * 0.3 + treble * 0.3)
            hue = (theta1 / (2*math.pi) + angle * 0.01) % 1.0
            r_c, g_c, b_c = neon_color(intensity, hue, 1.0)
            
            glColor3f(r_c, g_c, b_c)
    
            glVertex3fv(points[0])
            glVertex3fv(points[1])
            glVertex3fv(points[0])
            glVertex3fv(points[2])
    
    glEnd()

def draw_torus_rays(R, r, angle, bass, mid, treble):
    if bass < 0.2 and treble < 0.3:
        return
    
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE)
    glLineWidth(1.5 + bass*1)
    
    num_rays = 12 + int(bass * 12)
    ray_length = 2.0 + bass * 8
    
    glBegin(GL_LINES)
    
    for i in range(num_rays):
        theta = 2 * math.pi * i / num_rays
        phi = math.pi * 0.5
        

        x = (R + r * math.cos(phi)) * math.cos(theta + angle * 0.01)
        y = (R + r * math.cos(phi)) * math.sin(theta + angle * 0.01)
        z = r * math.sin(phi)
        

        dir_x = math.cos(theta) * 0.7
        dir_y = math.sin(theta) * 0.7
        dir_z = 0.3
        

        hue = (theta / (2*math.pi) + angle * 0.005) % 1.0
        intensity = 0.5 + 0.5 * (bass if i % 2 else treble)
        r_c, g_c, b_c = neon_color(intensity, hue, 1.2)
        

        glColor3f(r_c, g_c, b_c)
        glVertex3f(x, y, z)
        glVertex3f(
            x + dir_x * ray_length,
            y + dir_y * ray_length,
            z + dir_z * ray_length
        )
        

        if bass > 0.4:
            glColor4f(r_c*0.7, g_c*0.7, b_c*0.7, 0.6)
            glVertex3f(x, y, z)
            glVertex3f(
                x + dir_x * ray_length * 0.6,
                y + dir_y * ray_length * 0.6,
                z + dir_z * ray_length * 0.6
            )
    
    glEnd()
    glDisable(GL_BLEND)

def draw_spiral(vertices, edges, angle, bass, mid, treble):
    glLineWidth(1.0 + treble*3)
    glBegin(GL_LINES)
    
    n = len(vertices)

    for i in range(n-1):
        t = i / (n-1)
        intensity = 0.3 + 0.7 * (bass * 0.4 + mid * 0.3 + treble * 0.3)
        hue = (t + angle * 0.01) % 1.0
        r_c, g_c, b_c = neon_color(intensity, hue, 1.0)
        
        glColor3f(r_c, g_c, b_c)
        
        glVertex3fv(vertices[i])
        glVertex3fv(vertices[i+1])
        
        if i % 5 == 0 and bass > 0.3:
            glVertex3fv(vertices[i])
            glVertex3fv((vertices[i][0]*1.2, vertices[i][1]*1.2, vertices[i][2]))
    
    glEnd()

def draw_vortex(vertices, edges, angle, bass, mid, treble):
    glLineWidth(1.0 + treble*4)
    glBegin(GL_LINES)
    
    n = len(vertices)

    for i in range(n-1):
        t = i / (n-1)

        intensity = 0.4 + 0.6 * (bass * 0.2 + mid * 0.3 + treble * 0.5)
        hue = (t + angle * 0.02 + treble * 0.2) % 1.0
        r_c, g_c, b_c = neon_color(intensity, hue, 1.2)
        
        glColor3f(r_c, g_c, b_c)
        
        glVertex3fv(vertices[i])
        glVertex3fv(vertices[i+1])
        

        if i % 10 == 0 and (mid > 0.3 or treble > 0.4):
            j = min(i + random.randint(5, 20), n-1)
            glVertex3fv(vertices[i])
            glVertex3fv(vertices[j])
    
    glEnd()

def draw_eye(vertices, edges, angle, bass, mid, treble):
    glLineWidth(1.0 + treble*3)
    glBegin(GL_LINES)
    
    n = len(vertices)
    iris_start = n//2  
    

    for e in edges:
        if e[0] < iris_start and e[1] < iris_start:
            v1 = vertices[e[0]]
            v2 = vertices[e[1]]
            
            intensity = 0.5 + 0.5 * bass
            hue = (angle * 0.01) % 1.0
            r, g, b = neon_color(intensity, hue, 1.0)
            
            glColor3f(r, g, b)
            glVertex3fv(v1)
            glVertex3fv(v2)
    
 
    pupil_center = n - 1  
    for e in edges:
        if e[0] >= iris_start or e[1] >= iris_start:
            v1 = vertices[e[0]]
            v2 = vertices[e[1]]
            
            if e[0] == pupil_center or e[1] == pupil_center:

                glColor3f(0, 0, 0)
            else:

                intensity = 0.7 + 0.3 * mid
                hue = (angle * 0.02 + 0.5) % 1.0  
                r, g, b = neon_color(intensity, hue, 1.0)
                glColor3f(r, g, b)
            
            glVertex3fv(v1)
            glVertex3fv(v2)
    
    glEnd()

def draw_rays(base_vertices, angle, intensity, current_scale=1.0):
    if intensity < 0.1:
        return
        
    scale_factor = current_scale * (1.0 + intensity * 3.0)
    
    glLineWidth(1.0 + intensity*1)
    glBegin(GL_LINES)
    
    for i, v in enumerate(base_vertices):
        hue = (angle*0.01 + i/len(base_vertices)) % 1.0
        r, g, b = neon_color(intensity*0.7, hue, 0.8)
        glColor3f(r, g, b)
        
        start_point = (v[0]*current_scale, v[1]*current_scale, v[2]*current_scale)
        end_point = (
            v[0] * scale_factor * (1.0 + intensity * 0.5),
            v[1] * scale_factor * (1.0 + intensity * 0.5),
            v[2] * scale_factor * (1.0 + intensity * 0.5)
        )
        
        glVertex3fv(start_point)
        glVertex3fv(end_point)
    
    glEnd()

def interpolate_vertices(v1, v2, t):
    if not v1 or not v2:
        return []
    
    t_ease = t * t * (3 - 2 * t)
    return [(
        (1-t_ease)*x1 + t_ease*x2,
        (1-t_ease)*y1 + t_ease*y2,
        (1-t_ease)*z1 + t_ease*z2
    ) for (x1,y1,z1),(x2,y2,z2) in zip(v1,v2)]

def equalize_vertices(v_from, v_to):
    len_from, len_to = len(v_from), len(v_to)
    if len_from < len_to:
        return v_from + [v_from[-1]]*(len_to-len_from), v_to
    if len_to < len_from:
        return v_from, v_to + [v_to[-1]]*(len_from-len_to)
    return v_from, v_to

def draw_wireframe(vertices, edges, angle, bass, mid, treble, shape_name):
    glLineWidth(1.0 + treble*3)
    glBegin(GL_LINES)
    
    base_scale = scale_factors.get(shape_name, 1.0)
    scale_mod = 1.0 + bass * 0.5
    
    n = len(vertices)
    for e in edges:
        for i in range(len(e)-1):
            idx1, idx2 = e[i], e[i+1]
            if idx1 < n and idx2 < n:
                t = i / len(edges)
                intensity = 0.3 + 0.7 * (bass * 0.4 + mid * 0.3 + treble * 0.3)
                hue = (t + angle * 0.01) % 1.0
                r_c, g_c, b_c = neon_color(intensity, hue, 1.0)
                
                glColor3f(r_c, g_c, b_c)
                
                v1 = (vertices[idx1][0]*base_scale*scale_mod, 
                      vertices[idx1][1]*base_scale*scale_mod, 
                      vertices[idx1][2]*base_scale*scale_mod)
                v2 = (vertices[idx2][0]*base_scale*scale_mod, 
                      vertices[idx2][1]*base_scale*scale_mod, 
                      vertices[idx2][2]*base_scale*scale_mod)
                
                glVertex3fv(v1)
                glVertex3fv(v2)
    
    glEnd()

def get_current_shape_data(t, idx_from, idx_to):
    if shapes[idx_from][0] == "torus" or shapes[idx_to][0] == "torus" or \
       shapes[idx_from][0] == "spiral" or shapes[idx_to][0] == "spiral" or \
       shapes[idx_from][0] == "vortex" or shapes[idx_to][0] == "vortex":
        return [], []
    
    v1 = shapes[idx_from][1]
    v2 = shapes[idx_to][1]
    v1_eq, v2_eq = equalize_vertices(v1, v2)
    verts = interpolate_vertices(v1_eq, v2_eq, t)
    
    edges = shapes[idx_to][2]
    
    return verts, edges

def main_loop():
    global audio_buffer

    pygame.init()
    pygame.display.set_caption("yoobilizer")
    screen = pygame.display.set_mode((1280, 720), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    width, height = screen.get_size()

    font = init_font()
    
    glEnable(GL_DEPTH_TEST)
    gluPerspective(60, width/height, 0.1, 100.0)
    glTranslatef(0, 0, -50)

    audio_analyzer = AudioAnalyzer(SAMPLE_RATE, BUFFER_SIZE)

    try:
        device_id = find_device('loopback') or find_device('blackhole')
        if device_id is None:
            print("could not find loopback/blackhole, using default input device")
            device_id = None
        stream = sd.InputStream(device=device_id, channels=1, callback=audio_callback,
                              blocksize=BUFFER_SIZE, samplerate=SAMPLE_RATE)
        stream.start()
    except Exception as e:
        print(f"audio error: {e}")
        stream = None

    idx = 0
    next_idx = random.randint(1, len(shapes)-1)
    transition_dur = 1
    t_start = time.time()
    angle = 0
    torus_angle = 0
    spiral_angle = 0
    vortex_angle = 0
    camera_distance = 50
    camera_angle_x = 0
    camera_angle_y = 0
    last_shape_change = time.time()

    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f and (pygame.key.get_mods() & pygame.KMOD_ALT):
                pygame.display.toggle_fullscreen()
                width, height = pygame.display.get_surface().get_size()
                glViewport(0, 0, width, height)
                gluPerspective(60, width/height, 0.1, 175.0)

        now = time.time()
        t = min((now - t_start) / transition_dur, 2.0)
        
        analysis = audio_analyzer.analyze(audio_buffer)
        bass = analysis['bass']
        mid = analysis['mid']
        treble = analysis['treble']
        energy = analysis['energy']
        
        glClearColor(0.05, 0.05, 0.1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        camera_angle_x += 0.1 * energy + 0.05 * bass
        camera_angle_y += 0.07 * energy + 0.03 * mid
        camera_distance = 50 - 5 * bass + 3 * treble
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60 + 10 * energy, width/height, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            camera_distance * math.sin(camera_angle_x),
            camera_distance * math.sin(camera_angle_y),
            camera_distance * math.cos(camera_angle_x) * math.cos(camera_angle_y),
            0, 0, 0,
            0, 1, 0
        )

        glPushMatrix()
        

        rotation_speed = 0.5 + bass * 0.5 + mid * 0.3 + treble * 0.2
        angle += rotation_speed * 0.040
        glRotatef(angle, 0.5, 1.0, 0.3)
        

        scale = 0.5 + bass * 1.5 + mid * 0.5
        glScalef(scale, scale, scale)

        current_shape = shapes[idx][0]
        next_shape = shapes[next_idx][0] 
        
        if current_shape == "torus" or next_shape == "torus":
            if t < 0.5:
                verts, edges = get_current_shape_data(t*2, idx, idx)
                if verts:
                    draw_wireframe(verts, edges, angle, bass, mid, treble, current_shape)
            elif t > 0.5:
                verts, edges = get_current_shape_data((t-0.5)*2, next_idx, next_idx)
                if verts:
                    draw_wireframe(verts, edges, angle, bass, mid, treble, next_shape)
            
            torus_angle += 2 + bass*10
            draw_torus(R=5, r=1.5, N=50, n=20, angle=torus_angle, 
                      bass=bass, mid=mid, treble=treble)
            draw_torus_rays(R=5, r=1.5, angle=torus_angle, bass=bass, mid=mid, treble=treble)
        
        elif current_shape == "spiral" or next_shape == "spiral":
            if t < 0.5:
                verts, edges = get_current_shape_data(t*2, idx, idx)
                if verts:
                    draw_wireframe(verts, edges, angle, bass, mid, treble, current_shape)
            elif t > 0.5:
                verts, edges = get_current_shape_data((t-0.5)*2, next_idx, next_idx)
                if verts:
                    draw_wireframe(verts, edges, angle, bass, mid, treble, next_shape)
            
            spiral_angle += 5 + bass*10
            draw_spiral(spiral_verts, spiral_edges, spiral_angle, bass, mid, treble)
        
        elif current_shape == "vortex" or next_shape == "vortex":
            if t < 0.5:
                verts, edges = get_current_shape_data(t*2, idx, idx)
                if verts:
                    draw_wireframe(verts, edges, angle, bass, mid, treble, current_shape)
            elif t > 0.5:
                verts, edges = get_current_shape_data((t-0.5)*2, next_idx, next_idx)
                if verts:
                    draw_wireframe(verts, edges, angle, bass, mid, treble, next_shape)
            
            vortex_angle += 8 + treble*15
            draw_vortex(vortex_verts, vortex_edges, vortex_angle, bass, mid, treble)
        
        elif current_shape == "eye" or next_shape == "eye":
            if t < 0.5:
                verts, edges = get_current_shape_data(t*2, idx, idx)
                if verts:
                    if current_shape == "eye":
                        draw_eye(verts, edges, angle, bass, mid, treble)
                    else:
                        draw_wireframe(verts, edges, angle, bass, mid, treble, current_shape)
            elif t > 0.5:
                verts, edges = get_current_shape_data((t-0.5)*2, next_idx, next_idx)
                if verts:
                    if next_shape == "eye":
                        draw_eye(verts, edges, angle, bass, mid, treble)
                    else:
                        draw_wireframe(verts, edges, angle, bass, mid, treble, next_shape)
        
        else:
            verts, edges = get_current_shape_data(t, idx, next_idx)
            if verts:
                draw_wireframe(verts, edges, angle, bass, mid, treble, shapes[next_idx][0])
        
        if bass > 0.3:
            draw_rays(shapes[idx][1], angle, bass, scale)
        
        glPopMatrix()


        if energy > 0.1:
            draw_floating_chars(font, angle, energy, bass, mid, treble)


        if t >= 1.0 or (energy > 0.7 and t > 0.3):
            idx = next_idx
            next_idx = random.randint(0, len(shapes)-1)
            while next_idx == idx:
                next_idx = random.randint(0, len(shapes)-1)
            t_start = now
            transition_dur = 1.0 + random.random() * 1.0
            
        pygame.display.flip()
        clock.tick(60)

    if stream:
        stream.stop()
        stream.close()
    pygame.quit()

if __name__ == "__main__":
    main_loop()
