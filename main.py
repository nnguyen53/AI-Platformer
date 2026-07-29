import pygame
import sys
import pygame_widgets
from pygame_widgets.dropdown import Dropdown
from pygame_widgets.slider import Slider
from game.menu import *
from game.modes import *
from agent.models import MODELS
from game.maps import MAPS

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Floor is Lava")
    clock = pygame.time.Clock()
    selection_screen = 1

    menu_buttons = {  # main menu buttons
        "Train_AI": pygame.Rect(300, 220, 200, 60),
        "Play_Manual": pygame.Rect(300, 320, 200, 60),
        "Quit_Game": pygame.Rect(300, 420, 200, 60),
    }

    training_buttons = {
        "Start_Training": pygame.Rect(300, 380, 200, 60)
    }

    dropdown_font = pygame.font.Font(GAME_FONT, 20)

    model_dropdown = Dropdown(screen, 300, 155, 140, 40, 
                              name="Untrained", 
                              choices=list(MODELS.keys()),
                              borderRadius=3,
                              colour=(100, 160, 220),
                              values=list(MODELS.values()),
                              direction='down',
                              textHAlign='centre',
                              font=dropdown_font)

    level_slider = Slider(screen, 300, 285, 300, 20,
                            min=1,
                            max=len(MAPS),
                            step=1,
                            initial=1,
                            colour=(255, 255, 255),
                            handleColour=(100, 160, 220),
                            valueColour=(110, 176, 242))

    model_dropdown.hide() # initially hide training config UI
    level_slider.hide()

    while True:  # main menu loop
        clock.tick(60)

        mouse_pos = pygame.mouse.get_pos()
        hovered_button = None

        buttons = menu_buttons if selection_screen == 1 else training_buttons # determine which buttons to used based on what screen is current

        for name, rect in buttons.items():
            if rect.collidepoint(mouse_pos): # detect cursor hover
                hovered_button = name
                break

        if selection_screen == 1:
            draw_menu(screen, menu_buttons, hovered_button)
        elif selection_screen == 2:
            draw_training_config(screen, model_dropdown, level_slider, training_buttons, hovered_button)

        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if selection_screen == 1:
                    if menu_buttons["Train_AI"].collidepoint(mouse_pos): # handle different button presses across screens
                        selection_screen = 2
                        model_dropdown.show()
                        level_slider.show()
                    elif menu_buttons["Play_Manual"].collidepoint(mouse_pos):
                        run_manual_mode(screen, clock)
                        selection_screen = 1
                    elif menu_buttons["Quit_Game"].collidepoint(mouse_pos):
                        pygame.quit()
                        sys.exit()
                elif selection_screen == 2:
                    if training_buttons["Start_Training"].collidepoint(mouse_pos):
                        model = model_dropdown.getSelected()
                        if not model:
                            model = MODELS["Untrained"]

                        level = level_slider.getValue()

                        model_dropdown.hide()
                        level_slider.hide()

                        run_ai_mode(screen, clock, model[0], level)
                        selection_screen = 1
        pygame_widgets.update(events)
        pygame.display.update()
        