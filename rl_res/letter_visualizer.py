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
CANVAS_SIZE = 280
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
WIN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("CNN Letter Visualizer")
FONT = pygame.font.SysFont("consolas", 18)

CANVAS = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
CANVAS.fill((255, 255, 255))
BRUSH_RADIUS = 12

# ------------------------
# CNN Network
# ------------------------
class CNNNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*14*14, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1,1,28,28)
        h1 = torch.relu(self.conv1(x))
        h2 = torch.relu(self.conv2(h1))
        x = self.pool(h2)
        flat = x.view(x.size(0), -1)
        h3 = torch.relu(self.fc1(flat))
        out = self.fc2(h3)
        return out, h3, h1, h2

# ------------------------
# Data & categories
# ------------------------
categories = {}
X = []
Y = []

net = CNNNet(1)  # will re-init on first category
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

def draw_feature_maps(surface, fmap, x0, y0, cell=40):
    fmap = fmap[0].detach().numpy()  # take first sample
    num = fmap.shape[0]
    for i in range(num):
        img = fmap[i]
        img = (img - img.min())/(img.max()-img.min()+1e-6)
        img = (img*255).astype(np.uint8)
        img3 = np.stack([img]*3, axis=2)
        s = pygame.surfarray.make_surface(img3)
        s = pygame.transform.scale(s, (cell,cell))
        surface.blit(s, (x0 + (i%8)*cell, y0 + (i//8)*cell))

def draw_output(surface, out, x0, y0):
    probs = torch.softmax(out, dim=1)[0].detach().numpy()
    for i, p in enumerate(probs):
        color = (int(p*255),0,int((1-p)*255))
        pygame.draw.rect(surface, color, (x0, y0+i*30, int(p*200),25))
        txt = FONT.render(f"{list(categories.keys())[i]}: {p*100:.1f}%", True, (0,0,0))
        surface.blit(txt, (x0+210, y0+i*30))

def draw_latent(surface, X, Y, x0, y0):
    if len(X)<2:
        return
    vecs = torch.stack(X).numpy()
    labels = np.array(Y)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(vecs)
    for (px, py), lbl in zip(proj, labels):
        color = (255,0,0) if lbl==0 else (0,255,0) if lbl==1 else (0,0,255)
        pygame.draw.circle(surface, color, (int(x0+px*50), int(y0+py*50)), 6)

# ------------------------
# Main loop
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
            if event.key == pygame.K_SPACE:
                CANVAS.fill((255,255,255))

            key = event.unicode.upper()
            if key and key not in categories:
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

    if pygame.mouse.get_pressed()[0]:
        x,y = pygame.mouse.get_pos()
        if x<CANVAS_SIZE:
            draw_circle((x,y))
    if pygame.mouse.get_pressed()[2]:
        x,y = pygame.mouse.get_pos()
        if x<CANVAS_SIZE:
            draw_circle((x,y), erase=True)

    if len(categories)>0:
        vec = preprocess(CANVAS)
        out, h3, h1, h2 = net(vec)
        draw_feature_maps(WIN,h1,CANVAS_SIZE+20,20,cell=30)
        draw_feature_maps(WIN,h2,CANVAS_SIZE+20,200,cell=20)
        draw_output(WIN,out,CANVAS_SIZE+20,400)
        draw_latent(WIN,X,Y,CANVAS_SIZE+400,20)

    instr = FONT.render("Draw | Type new category key to add | Press category key to label | SPACE clear | Q quit", True,(0,0,0))
    WIN.blit(instr,(20,CANVAS_SIZE+10))

    pygame.display.update()

pygame.quit()

