import directory_management as dm
import LoadingFootstep as load
import result as rs
from displayconfig import *


def upload(screen):
    file = dm.getfile("Select a shoe image")
    fade(screen, 360, 640)
    load.loading()
    fade(screen, 360, 640)
    return rs.result(file)

def livecamera():
    shoe_shape, shoe_color = rs.live()
    load.loading()
    return rs.displayresult(shoe_shape, shoe_color)

def main():
    pygame.init()

    size = (360, 640)
    screen = pygame.display.set_mode(size)
    icon = pygame.image.load("images/icon.gif")
    pygame.display.set_icon(icon)
    pygame.display.set_caption("sho.fi")
    background_image = pygame.image.load("images/bg.gif").convert()
    button = pygame.image.load("images/button.gif").convert()
    button_pressed = pygame.image.load("images/buttonpressed.gif").convert()

    camera = pygame.image.load("images/camera.gif")
    camera_pressed = pygame.image.load("images/camerapressed.gif")

    screen.blit(background_image, [0, 0])
    screen.blit(button, [30, 180])
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == QUIT:
                done = True
        pygame.display.flip()

        #hover & click
        mouse_position = pygame.mouse.get_pos()
        mouse_click = pygame.mouse.get_pressed()

        uploadButton = display_button(screen, button, button_pressed, 30, 170, 246, 241, mouse_position, mouse_click)
        cameraButton = display_button(screen, camera, camera_pressed, 127, 480, 96, 101, mouse_position, mouse_click)
        cameraInfo = message_display(screen, "or use webcam", white, 180, 470, "light_italic", 25, black)
        
        if uploadButton == True:
            if(upload(screen)==1):
                main()
        if cameraButton == True:
            if(livecamera()==1):
                main()
        
    pygame.quit()
main()