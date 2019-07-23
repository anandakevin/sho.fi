import pygame, sys
from pygame.locals import *
import predict as pred
from displayconfig import * 

def displayresult(shoe_shape, shoe_color):
    pygame.init()
    size = (360, 640)
    screen = pygame.display.set_mode(size)
    # pygame.display.set_caption("Shoe Identificator")
    background_image = pygame.image.load("images/resultbg.gif")
    screen.blit(background_image, [0, 0])
    done = False
    
    message_display(screen, shoe_color[0], white, 180, 270, "light", 35)
    message_display(screen, shoe_color[1]+" , "+shoe_color[2], white, 180, 300, "thin_italic", 20)
    ###################################
    message_display(screen, shoe_shape[0], black, 180, 340, "regular_italic", 40)
    ###################################
    message_display(screen, "alternative result:", white, 180, 380, "light_italic", 20)
    message_display(screen, shoe_shape[1], white, 180, 400, "thin_italic", 20)
    message_display(screen, shoe_shape[2], white, 180, 420, "thin_italic", 20)
    ###################################
    back = message_display(screen, "upload another shoe image", white, 180, 600, "light_italic", 20, black)

    while not done:
        for event in pygame.event.get():
            if event.type == QUIT:
                done = True
        pygame.display.update()

        mouse_position = pygame.mouse.get_pos()
        mouse_click = pygame.mouse.get_pressed()

        if back.collidepoint(mouse_position):
            message_display(screen, "upload another shoe image", purple, 180, 600, "light_italic", 20, black)
            if (mouse_click[0] == 1):
                 return 1
        else:
            message_display(screen, "upload another shoe image", white, 180, 600, "light_italic", 20, black)
    pygame.quit()

def live():
    return pred.live_processing()

def result(file):
    shoe_shape = pred.getprediction(file, 'image', 'shape')
    shoe_color = pred.getprediction(file, 'image', 'color')
    return displayresult(shoe_shape, shoe_color)