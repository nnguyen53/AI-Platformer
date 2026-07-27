import pygame
from utils.consts import GAME_FONT, GAME_FONT_BOLD
from utils.helpers import draw_multiline_text
from game.maps import MAPS
from agent.models import MODELS

def draw_button(surface, rect, label, hovered=False):  
    color = (70, 120, 180) if not hovered else (100, 160, 220)
    pygame.draw.rect(surface, color, rect, border_radius=8)
    pygame.draw.rect(surface, (255, 255, 255), rect, 3, border_radius=8)

    font = pygame.font.Font(GAME_FONT_BOLD, 28)
    text_surf = font.render(label, True, (255, 255, 255))
    text_rect = text_surf.get_rect(center=rect.center)
    surface.blit(text_surf, text_rect)


def draw_menu(screen, buttons, hovered_button):  
    screen.fill((12, 15, 24))

    title_font = pygame.font.Font(GAME_FONT_BOLD, 46)
    title_surf = title_font.render("FLOOR IS LAVA", True, (255, 255, 255))
    screen.blit(title_surf, (255, 90))

    subtitle_font = pygame.font.Font(GAME_FONT, 20)
    subtitle_surf = subtitle_font.render("Choose how you want to play", True, (200, 200, 200))
    screen.blit(subtitle_surf, (255, 150))

    for name, rect in buttons.items():
        draw_button(screen, rect, name.replace("_", " "), hovered=name == hovered_button)


def draw_training_config(screen, model_dropdown, level_slider, buttons, hovered_button):
    screen.fill((12, 15, 24))

    label_font = pygame.font.Font(GAME_FONT_BOLD, 30)
    description_font = pygame.font.Font(GAME_FONT, 20)

    title_font = pygame.font.Font(GAME_FONT_BOLD, 46)
    title_surf = title_font.render("Configure Training", True, (255, 255, 255))
    screen.blit(title_surf, (225, 90))

    model_label_surf = label_font.render("Select model:", True, (255, 255, 255))
    screen.blit(model_label_surf, (100, 160))

    model = model_dropdown.getSelected() 
    if not model:
        model = MODELS["Untrained"]

    draw_multiline_text(screen, model[1], description_font, (255, 255, 255), (100, 210))

    level_label_surf = label_font.render("Select level:", True, (255, 255, 255))
    screen.blit(level_label_surf, (100, 280))

    level = level_slider.getValue()
    draw_multiline_text(screen, MAPS[level]["description"], description_font, (255, 255, 255), (100, 330))

    for name, rect in buttons.items():
        draw_button(screen, rect, name.replace("_", " "), hovered=name == hovered_button)


def draw_pause_overlay(screen):  
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((10, 10, 15, 180))  # semi-transparent dark tint so the frozen game is still visible behind it
    screen.blit(overlay, (0, 0))

    title_font = pygame.font.Font(GAME_FONT_BOLD, 40)
    title_surf = title_font.render("PAUSED", True, (255, 255, 255))
    title_rect = title_surf.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 - 30))
    screen.blit(title_surf, title_rect)

    hint_font = pygame.font.Font(GAME_FONT, 22)
    hint_surf = hint_font.render("Press ESC to Resume  |  Press Q to Quit to Menu", True, (210, 210, 210))
    hint_rect = hint_surf.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 + 20))
    screen.blit(hint_surf, hint_rect)

    pygame.display.flip()