import pygame
from pygame.locals import *


def affichage(nb_col, nb_lin):
    pygame.init()

    # -------------------- variable
    desc = "test de pygame"
    taille = (1080, 720)

    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)

    quadrillage = (nb_col, nb_lin)              # taille du quadrillage
    taille_qua = 30                     # taille par carreau
    offset_qua = (40, 40)             # origine du quadrillage

    # -------------------- création de la fenêtre
    pygame.display.set_caption(desc)
    screen = pygame.display.set_mode(taille)  # -> Surface
    screen.fill(WHITE)
    pygame.display.flip()

    # -------------------- running
    running = True

    # -------------------- création quadrillage
    for i in range(quadrillage[0]):
        for j in range(quadrillage[1]):
            x = taille_qua*i + offset_qua[0]
            y = taille_qua*j + offset_qua[1]
            pygame.draw.rect(
                screen, BLACK, ((x, y), (taille_qua, taille_qua)), 1)
            # print(x, y, x+99, y+99)

    my_font = pygame.font.SysFont('Comic Sans MS', 30)
    text = font.render('GeeksForGeeks', True, green, blue)
    # -------------------- running
    while running:
        # pygame.event.get() -> list
        # x_min, y_min, x_max, y_max
        pygame.display.update()
        for event in pygame.event.get():
            # print(event)  # très utile !!!
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    screen.scroll(0, -10)
                if event.key == pygame.K_s:
                    screen.scroll(0, 10)

    pygame.quit()
