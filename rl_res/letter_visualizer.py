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



