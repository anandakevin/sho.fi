import pygame
from sys import exit

# cara install pygame:
# pip install pygame
# ATAU
# python -m install pygame

def loading():
    pygame.init() #inisialisasi. Kmrn lupa pakai gpp.

    size = (360, 640) #ukuran layar
    screen = pygame.display.set_mode(size) #masukkin ukurannya mau berapa

    img = pygame.image.load("images/left1.png").convert_alpha() #ini buat tampung image kaki kanan
    img1 = pygame.image.load("images/right1.png").convert_alpha() #ini buat tampung image kaki kiri
    img2 = pygame.image.load("images/left2.png").convert_alpha()
    img3 = pygame.image.load("images/right2.png").convert_alpha()
    img4 = pygame.image.load("images/left3.png").convert_alpha()
    img5 = pygame.image.load("images/right3.png").convert_alpha()
    img6 = pygame.image.load("images/left4.png").convert_alpha()
    img7 = pygame.image.load("images/right4.png").convert_alpha()

    load = pygame.image.load("images/loading(1).png").convert_alpha()
    titik = pygame.image.load("images/titik.png").convert_alpha()

    # while True: #kemarin buat loadingnya masih di-looping-in forever
    #     ev = pygame.event.poll() #buat tampung kl pencet exit
    #     if ev.type == pygame.QUIT: #dibuat kondisinya biar keluar looping
    #         break

    background_image = pygame.image.load("images/bg(1).gif").convert_alpha()
    screen.blit(background_image, [0, 0])

    screen.blit(load, (71, 570))
    screen.blit(img, ( 115, 450 ))
    pygame.display.update()
    pygame.time.wait(200)

    screen.blit(img1, ( 185, 390 ))
    pygame.display.update()
    pygame.time.wait(200)

    screen.blit(titik, (214, 590))
    screen.blit(img2, ( 115, 330 ))
    pygame.display.update()
    pygame.time.wait(500)

    screen.blit(img3, (185, 270))
    pygame.display.update()
    pygame.time.wait(300)

    screen.blit(titik, (229, 590))
    screen.blit(img4, (115, 210))
    pygame.display.update()
    pygame.time.wait(100)

    screen.blit(img5, (185, 150))
    pygame.display.update()
    pygame.time.wait(200)

    screen.blit(titik, (244, 590))
    screen.blit(img6, (115, 90))
    pygame.display.update()
    pygame.time.wait(100)

    screen.blit(img7, (185, 30))
    pygame.display.update()
    pygame.time.wait(100)

    # while True:  # kemarin buat loadingnya masih di-looping-in forever
    #     ev = pygame.event.poll() #buat tampung kl pencet exit
    #     if ev.type == pygame.QUIT: #dibuat kondisinya biar keluar looping
    #         pygame.quit()

    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             exit()
    return
