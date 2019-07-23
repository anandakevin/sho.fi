import pygame
from pygame.locals import *

white = pygame.Color(255,255,255)
black = pygame.Color(0,0,0)
purple = pygame.Color(214, 91, 202)

def message_display(screen, text, color, x, y, style, size, bg=None):
    textFont = pygame.font.Font("fonts/bold-italic.otf", size)
    if style == "bold_italic":
        textFont = pygame.font.Font("fonts/bold-italic.otf", size)
    if style == "black":
        textFont = pygame.font.Font("fonts/black.otf", size)
    if style == "light":
        textFont = pygame.font.Font("fonts/light.otf", size)
    if style == "light_italic":
        textFont = pygame.font.Font("fonts/light-italic.otf", size)
    if style == "medium_italic":
            textFont = pygame.font.Font("fonts/medium-italic.otf", size)
    if style == "regular_italic":
        textFont = pygame.font.Font("fonts/regular-italic.otf", size)
    if style == "thin_italic":
        textFont = pygame.font.Font("fonts/thin-italic.otf", size)
    textSurface = textFont.render(text, True, color, bg)
    textRect = textSurface.get_rect()
    textRect.center = (x, y)
    screen.blit(textSurface, textRect)
    pygame.display.update()
    return textRect

def display_button(screen, image, image_pressed, x, y, w, h, mouse_position, mouse_click):
    if x + w > mouse_position[0] > x and y + h > mouse_position[1] > y:
        screen.blit(image_pressed, [x, y])
        if (mouse_click[0] == 1):
            return True
    else:
        screen.blit(image, [x, y])

def redrawWindow(screen):
    #screen.fill(white)
    #pygame.draw.rect(screen, (255, 0, 0), (200,300,200,200), 0)
    pygame.draw.rect(screen, (0, 255, 0), (500, 500, 100, 200), 0)

def fade(screen, width, height):
    fade = pygame.Surface((width, height))

    #fade.fill((0,0,0))
    for alpha in range(0, 30):
        fade.set_alpha(alpha)
        #redrawWindow(screen)
        screen.blit(fade, (0,0))
        pygame.display.update()
        #pygame.time.delay(1)