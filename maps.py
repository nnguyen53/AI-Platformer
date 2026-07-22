import pygame

MAPS = {
            1: {
                "platforms": [pygame.Rect(50, 500, 180, 20), pygame.Rect(280, 420, 150, 20), pygame.Rect(400, 340, 150, 20)],
                "player_start": (100, 450),
                "star": pygame.Rect(540, 290, 30, 30)
            },
            2: {
                "platforms": [pygame.Rect(50, 520, 120, 20), pygame.Rect(220, 440, 120, 20), pygame.Rect(400, 480, 100, 20), pygame.Rect(590, 360, 160, 20)],
                "player_start": (80, 470),
                "star": pygame.Rect(620, 310, 30, 30)
            },
            3: {
                "platforms": [pygame.Rect(50, 500, 180, 20), pygame.Rect(280, 420, 150, 20), pygame.Rect(400, 340, 150, 20), pygame.Rect(230, 280, 100, 20)],
                "player_start": (100, 450),
                "star": pygame.Rect(250, 220, 30, 30)
            },
            4: {
                "platforms": [
                    pygame.Rect(50, 550, 150, 20), 
                    pygame.Rect(250, 420, 100, 20), 
                    pygame.Rect(100, 250, 120, 20), 
                    pygame.Rect(300, 175, 200, 20), 
                    pygame.Rect(380, 350, 30, 20) 
                ],
                "player_start": (70, 500),
                "star": pygame.Rect(550, 120, 30, 30)
            },
            5: {
                # Easy, mostly flat — good low end of difficulty range
                "platforms": [
                    pygame.Rect(50, 520, 160, 20),
                    pygame.Rect(260, 480, 140, 20),
                    pygame.Rect(460, 440, 140, 20),
                    pygame.Rect(280, 380, 100, 20),
                ],
                "player_start": (100, 470),
                "star": pygame.Rect(310, 320, 30, 30)
            },
            6: {
                # Steeper, narrower platforms — tests vertical judgment + precision landing
                "platforms": [
                    pygame.Rect(60, 530, 150, 20),
                    pygame.Rect(80, 440, 90, 20),
                    pygame.Rect(220, 370, 80, 20),
                    pygame.Rect(180, 280, 70, 20),
                ],
                "player_start": (110, 480),
                "star": pygame.Rect(200, 220, 30, 30)
            },
            7: {
                # Zigzag with alternating left/right — tests horizontal direction switching
                "platforms": [
                    pygame.Rect(40, 520, 150, 20),
                    pygame.Rect(280, 460, 120, 20),
                    pygame.Rect(80, 390, 100, 20),
                    pygame.Rect(320, 320, 120, 20),
                ],
                "player_start": (90, 470),
                "star": pygame.Rect(360, 260, 30, 30)
            },
            8: {
                # Branching: two viable routes from platform 1 to the goal
                "platforms": [
                    pygame.Rect(50, 530, 170, 20),
                    pygame.Rect(260, 460, 100, 20),   # route A - lower path
                    pygame.Rect(80, 400, 90, 20),      # route B - upper path
                    pygame.Rect(280, 340, 100, 20),    # both routes converge here
                ],
                "player_start": (100, 480),
                "star": pygame.Rect(310, 280, 30, 30)
            },
            9: {
                # Longer chain, similar difficulty per-jump to level 4 but more platforms overall
                "platforms": [
                    pygame.Rect(40, 540, 150, 20),
                    pygame.Rect(240, 480, 110, 20),
                    pygame.Rect(90, 410, 100, 20),
                    pygame.Rect(280, 350, 100, 20),
                    pygame.Rect(150, 280, 90, 20),
                ],
                "player_start": (90, 490),
                "star": pygame.Rect(180, 220, 30, 30)
            },
            10: {
                # Wide flat gap, minimal height change — different jump "shape" than the others
                "platforms": [
                    pygame.Rect(50, 500, 160, 20),
                    pygame.Rect(280, 500, 130, 20),
                    pygame.Rect(480, 470, 130, 20),
                    pygame.Rect(300, 400, 100, 20),
                ],
                "player_start": (100, 450),
                "star": pygame.Rect(330, 340, 30, 30)
            },
            11: {
                "platforms": [pygame.Rect(600,460,150,20), pygame.Rect(420,500,120,20), pygame.Rect(250,460,120,20), pygame.Rect(80,500,140,20)],
                "player_start": (650,410),
                "star": pygame.Rect(110,450,30,30)
            },
            12: {
                "platforms": [pygame.Rect(80,280,150,20), pygame.Rect(300,340,120,20)],
                "player_start": (120,230),
                "star": pygame.Rect(460,380,30,30)
            },
            13: {
                "platforms": [pygame.Rect(350,240,140,20), pygame.Rect(180,300,110,20), pygame.Rect(360,370,100,20)],
                "player_start": (390,190),
                "star": pygame.Rect(390,320,30,30)
            },
            14: {
                "platforms": [pygame.Rect(550,530,150,20), pygame.Rect(400,460,90,20), pygame.Rect(500,340,80,20), pygame.Rect(340,290,90,20)],
                "player_start": (590,480),
                "star": pygame.Rect(390,230,30,30)
            },
            15: {
                "platforms": [pygame.Rect(50,540,160,20), pygame.Rect(260,480,100,20), pygame.Rect(90,420,90,20), pygame.Rect(280,360,100,20)],
                "player_start": (90,490),
                "star": pygame.Rect(310,300,30,30)
            },
        }