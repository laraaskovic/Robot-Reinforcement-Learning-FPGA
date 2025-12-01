'''
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

pygame.init()

# ------------------------
# Screen
# ------------------------
CANVAS_SIZE = 280
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 500
WIN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Dynamic Category Neural Net Visualizer")
FONT = pygame.font.SysFont("consolas", 18)

CANVAS = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
CANVAS.fill((255,255,255))
BRUSH_RADIUS = 12

# ------------------------
# Network
# ------------------------
class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 32)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        return out, h

# Start with empty categories
categories = {}   # key -> index
X = []
Y = []

net = SimpleNet(1)  # will reinitialize when first category added
optimizer = None
loss_fn = nn.CrossEntropyLoss()

# ------------------------
# Helpers
# ------------------------
def preprocess(surface):
    small = pygame.transform.scale(surface, (28,28))
    arr = pygame.surfarray.array3d(small).mean(axis=2)
    arr = 1 - arr/255.0
    return torch.tensor(arr.flatten(), dtype=torch.float32)

def draw_circle(pos, erase=False):
    color = (255,255,255) if erase else (0,0,0)
    pygame.draw.circle(CANVAS, color, pos, BRUSH_RADIUS)

def draw_hidden(surface, h, x0, y0):
    neurons = min(len(h),32)
    cell = 20
    for i in range(neurons):
        val = h[i].item()
        norm = (val - h.min().item())/(h.max().item()-h.min().item()+1e-6)
        gray = int(norm*255)
        pygame.draw.rect(surface,(gray,gray,255),(x0+(i%8)*cell, y0+(i//8)*cell, cell, cell))

def draw_output(surface, out, x0, y0):
    probs = torch.softmax(out,dim=0).detach().numpy()
    for i, p in enumerate(probs):
        color = (int(p*255),0,int((1-p)*255))
        pygame.draw.rect(surface, color, (x0, y0+i*30, int(p*200),25))
        txt = FONT.render(f"{list(categories.keys())[i]}: {p*100:.1f}%", True, (0,0,0))
        surface.blit(txt, (x0+210, y0+i*30))

# ------------------------
# Main Loop
# ------------------------
run = True
drawing = False

while run:
    WIN.fill((230,230,230))
    WIN.blit(CANVAS,(0,0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run=False

        if event.type == pygame.MOUSEBUTTONDOWN:
            drawing=True
        if event.type == pygame.MOUSEBUTTONUP:
            drawing=False

        if event.type == pygame.KEYDOWN:
            # Clear canvas
            if event.key == pygame.K_SPACE:
                CANVAS.fill((255,255,255))
            
            # Add new category dynamically
            key = event.unicode.upper()
            if key and key not in categories:
                categories[key] = len(categories)
                net = SimpleNet(len(categories))
                optimizer = optim.Adam(net.parameters(), lr=0.002)
                print(f"Added new category: {key}")

            # Label current drawing
            if key in categories:
                label = categories[key]
                X.append(preprocess(CANVAS))
                Y.append(label)

                # Train network if enough samples
                if len(Y) > 2:
                    data = torch.stack(X)
                    labels = torch.tensor(Y)
                    optimizer.zero_grad()
                    logits,_ = net(data)
                    loss = loss_fn(logits,labels)
                    loss.backward()
                    optimizer.step()
                    print(f"Trained on {len(Y)} samples | Loss: {loss.item():.4f}")

            # Quit
            if event.key == pygame.K_q:
                run=False

    if pygame.mouse.get_pressed()[0]:
        x,y = pygame.mouse.get_pos()
        if x<CANVAS_SIZE:
            draw_circle((x,y))
    if pygame.mouse.get_pressed()[2]:
        x,y = pygame.mouse.get_pos()
        if x<CANVAS_SIZE:
            draw_circle((x,y), erase=True)

    # Forward pass
    if len(categories)>0:
        vec = preprocess(CANVAS)
        out,h = net(vec)
        draw_hidden(WIN,h,CANVAS_SIZE+20,20)
        draw_output(WIN,out,CANVAS_SIZE+20,200)

    # Instructions
    instr = FONT.render("Draw | Type new category key to add | Press category key to label | SPACE clear | Q quit", True,(0,0,0))
    WIN.blit(instr,(20,CANVAS_SIZE+10))

    pygame.display.update()

pygame.quit()
'''





import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA

# ------------------------
# Pygame setup
# ------------------------
pygame.init()
CANVAS_SIZE = 320
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900

WIN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Neon CNN Letter Visualizer")

FONT = pygame.font.SysFont("consolas", 20, bold=True)

# Dark theme colors
BG = (10, 10, 25)
PANEL = (20, 20, 40)
NEON_BLUE = (80, 170, 255)
WHITE = (230, 230, 255)

# Canvas
CANVAS = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
CANVAS.fill((255, 255, 255))
BRUSH_RADIUS = 14

# ------------------------
# CNN Network
# ------------------------
class CNNNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*14*14, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1,1,28,28)
        h1 = torch.relu(self.conv1(x))
        h2 = torch.relu(self.conv2(h1))
        pooled = self.pool(h2)
        flat = pooled.view(pooled.size(0), -1)
        h3 = torch.relu(self.fc1(flat))
        out = self.fc2(h3)
        return out, h3, h1, h2

# ------------------------
# Data & categories
# ------------------------
categories = {}
X = []
Y = []

net = CNNNet(1)
optimizer = None
loss_fn = nn.CrossEntropyLoss()

# ------------------------
# Helpers
# ------------------------
def preprocess(surface):
    small = pygame.transform.scale(surface, (28,28))
    arr = pygame.surfarray.array3d(small).mean(axis=2)
    arr = 1 - arr/255.0
    return torch.tensor(arr.flatten(), dtype=torch.float32)

def draw_circle(pos, erase=False):
    color = (255,255,255) if erase else (0,0,0)
    pygame.draw.circle(CANVAS, color, pos, BRUSH_RADIUS)

# --- Draw feature maps cleanly in a grid
def draw_feature_grid(surface, fmap, x0, y0, cell):
    fmap = fmap[0].detach().numpy()
    C, H, W = fmap.shape

    cols = 8
    rows = int(np.ceil(C / cols))

    for i in range(C):
        img = fmap[i]
        img = (img - img.min())/(img.max()-img.min()+1e-6)
        img = (img*255).astype(np.uint8)
        img3 = np.stack([img, img*0.7, 255-img], axis=2)  # nice blueish tint

        s = pygame.surfarray.make_surface(img3)
        s = pygame.transform.scale(s, (cell,cell))

        cx = x0 + (i % cols)*cell
        cy = y0 + (i // cols)*cell
        pygame.draw.rect(surface, PANEL, (cx-2,cy-2,cell+4,cell+4))
        surface.blit(s, (cx, cy))

# --- Output bars
def draw_output(surface, out, x0, y0):
    probs = torch.softmax(out, dim=1)[0].detach().numpy()
    keys = list(categories.keys())

    for i, p in enumerate(probs):
        # neon blue to pink gradient
        r = int(60 + p*180)
        g = int(120 + p*50)
        b = int(255)

        w = int(p*250)

        pygame.draw.rect(surface, (r,g,b), (x0, y0+i*40, w, 30))
        pygame.draw.rect(surface, WHITE, (x0, y0+i*40, 250, 30), 2)

        txt = FONT.render(f"{keys[i]}: {p*100:.1f}%", True, WHITE)
        surface.blit(txt, (x0 + 270, y0 + i*40))

# --- Latent space PCA
def draw_latent(surface, X, Y, x0, y0):
    if len(X)<3:
        return

    vecs = torch.stack(X).numpy()
    labels = np.array(Y)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(vecs)

    for (px, py), lbl in zip(proj, labels):
        cx = int(x0 + px*60)
        cy = int(y0 + py*60)

        color = (80,160,255) if lbl==0 else (255,80,180) if lbl==1 else (80,255,150)
        pygame.draw.circle(surface, color, (cx,cy), 8)
        pygame.draw.circle(surface, WHITE, (cx,cy), 8, 2)

# ------------------------
# Main loop
# ------------------------
run = True
drawing = False

while run:
    WIN.fill(BG)

    # Draw canvas panel
    pygame.draw.rect(WIN, PANEL, (10,10,CANVAS_SIZE+20,CANVAS_SIZE+20))
    WIN.blit(CANVAS,(20,20))

    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run=False

        if event.type == pygame.MOUSEBUTTONDOWN:
            drawing=True
        if event.type == pygame.MOUSEBUTTONUP:
            drawing=False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                CANVAS.fill((255,255,255))

            key = event.unicode.upper()

            if key and key not in categories and key.isalpha():
                categories[key] = len(categories)
                net = CNNNet(len(categories))
                optimizer = optim.Adam(net.parameters(), lr=0.002)
                print(f"Added category: {key}")

            if key in categories:
                label = categories[key]

                X.append(preprocess(CANVAS))
                Y.append(label)

                if len(Y)>2:
                    data = torch.stack(X)
                    labels = torch.tensor(Y)
                    optimizer.zero_grad()
                    logits,_,_,_ = net(data)
                    loss = loss_fn(logits, labels)
                    loss.backward()
                    optimizer.step()
                    print(f"Trained on {len(Y)} samples | Loss: {loss.item():.4f}")

            if event.key == pygame.K_q:
                run=False

    # Drawing
    if drawing and pygame.mouse.get_pressed()[0]:
        x,y = pygame.mouse.get_pos()
        if 20 < x < 20+CANVAS_SIZE:
            draw_circle((x-20,y-20))

    if pygame.mouse.get_pressed()[2]: # erase
        x,y = pygame.mouse.get_pos()
        if 20 < x < 20+CANVAS_SIZE:
            draw_circle((x-20,y-20), erase=True)

    # If categories exist, show activations
    if len(categories)>0:
        vec = preprocess(CANVAS)
        out, h3, h1, h2 = net(vec)

        draw_feature_grid(WIN, h1, 380, 30, 60)
        draw_feature_grid(WIN, h2, 380, 350, 45)
        draw_output(WIN, out, 1000, 50)
        draw_latent(WIN, X, Y, 1050, 350)

    instr = FONT.render("Draw | Add new category by typing a letter | SPACE = Clear | Q = Quit", True, NEON_BLUE)
    WIN.blit(instr, (20, CANVAS_SIZE+40))

    pygame.display.update()

pygame.quit()
