import pygame

MAPS = {
            1: {
                # was level 15 - peak 98%, overall 84%
                "platforms": [pygame.Rect(50,540,160,20), pygame.Rect(260,480,100,20), pygame.Rect(90,420,90,20), pygame.Rect(280,360,100,20)],
                "player_start": (90,490),
                "star": pygame.Rect(310,300,30,30),
                "description": "Level 1 - Easy. Our best model achieved a 98% peak win rate here.",
                "peak_win_rate": 0.98,
            },
            2: {
                # was level 9 - peak 98%, overall 75%
                "platforms": [
                    pygame.Rect(40,540,150,20),
                    pygame.Rect(240,480,110,20),
                    pygame.Rect(90,410,100,20),
                    pygame.Rect(280,350,100,20),
                    pygame.Rect(150,280,90,20),
                ],
                "player_start": (90,490),
                "star": pygame.Rect(180,220,30,30),
                "description": "Level 2 - Easy. Our best model achieved a 98% peak win rate here.",
                "peak_win_rate": 0.98,
            },
            3: {
                # was level 8 - peak 96%, overall 86%
                "platforms": [
                    pygame.Rect(50,530,170,20),
                    pygame.Rect(260,460,100,20),  
                    pygame.Rect(80,400,90,20),      
                    pygame.Rect(280,340,100,20),   
                ],
                "player_start": (100,480),
                "star": pygame.Rect(310,280,30,30),
                "description": "Level 3 - Easy. Our best model achieved a 96% peak win rate here.",
                "peak_win_rate": 0.96,
            },
            4: {
                # was level 10 - peak 94%, overall 86%
                "platforms": [
                    pygame.Rect(50,500,160,20),
                    pygame.Rect(280,500,130,20),
                    pygame.Rect(480,470,130,20),
                    pygame.Rect(300,400,100,20),
                ],
                "player_start": (100,450),
                "star": pygame.Rect(330,340,30,30),
                "description": "Level 4 - Easy. Our best model achieved a 94% peak win rate here.",
                "peak_win_rate": 0.94,
            },
            5: {
                # was level 5 - peak 94%, overall 81%
                "platforms": [
                    pygame.Rect(50,520,160,20),
                    pygame.Rect(260,480,140,20),
                    pygame.Rect(460,440,140,20),
                    pygame.Rect(280,380,100,20),
                ],
                "player_start": (100,470),
                "star": pygame.Rect(310,320,30,30),
                "description": "Level 5 - Easy. Our best model achieved a 94% peak win rate here.",
                "peak_win_rate": 0.94,
            },
            6: {
                # was level 1 - peak 94%, overall 66%
                "platforms": [pygame.Rect(50,500,180,20), pygame.Rect(280,420,150,20), pygame.Rect(400,340,150,20)],
                "player_start": (100,450),
                "star": pygame.Rect(540,290,30,30),
                "description": "Level 6 - Easy. Our best model achieved a 94% peak win rate here.",
                "peak_win_rate": 0.94,
            },
            7: {
                # was level 11 - peak 78%, overall 42%
                "platforms": [pygame.Rect(600,460,150,20), pygame.Rect(420,500,120,20), pygame.Rect(250,460,120,20), pygame.Rect(80,500,140,20)],
                "player_start": (650,410),
                "star": pygame.Rect(110,450,30,30),
                "description": "Level 7 - Medium. Our best model achieved a 78% peak win rate here.",
                "peak_win_rate": 0.78,
            },
            8: {
                # was level 2 - peak 76%, overall 37%
                "platforms": [pygame.Rect(50,520,120,20), pygame.Rect(220,440,120,20), pygame.Rect(400,480,100,20), pygame.Rect(590,360,160,20)],
                "player_start": (80,470),
                "star": pygame.Rect(620,310,30,30),
                "description": "Level 8 - Medium. Our best model achieved a 76% peak win rate here.",
                "peak_win_rate": 0.76,
            },
            9: {
                # was level 14 - peak 74%, overall 31%
                "platforms": [pygame.Rect(550,530,150,20), pygame.Rect(400,460,90,20), pygame.Rect(500,340,80,20), pygame.Rect(340,290,90,20)],
                "player_start": (590,480),
                "star": pygame.Rect(390,230,30,30),
                "description": "Level 9 - Medium. Our best model achieved a 74% peak win rate here.",
                "peak_win_rate": 0.74,
            },
            10: {
                # was level 13 - peak 70%, overall 55%
                "platforms": [pygame.Rect(350,240,140,20), pygame.Rect(180,300,110,20), pygame.Rect(360,370,100,20)],
                "player_start": (390,190),
                "star": pygame.Rect(390,320,30,30),
                "description": "Level 10 - Medium. Our best model achieved a 70% peak win rate here.",
                "peak_win_rate": 0.70,
            },
            11: {
                # was level 12 - peak 70%, overall 38%
                "platforms": [pygame.Rect(80,280,150,20), pygame.Rect(300,340,120,20)],
                "player_start": (120,230),
                "star": pygame.Rect(460,380,30,30),
                "description": "Level 11 - Medium. Our best model achieved a 70% peak win rate here.",
                "peak_win_rate": 0.70,
            },
            12: {
                # was level 3 - peak 70%, overall 19%
                "platforms": [pygame.Rect(50,500,180,20), pygame.Rect(280,420,150,20), pygame.Rect(400,340,150,20), pygame.Rect(230,280,100,20)],
                "player_start": (100,450),
                "star": pygame.Rect(250,220,30,30),
                "description": "Level 12 - Medium. Our best model achieved a 70% peak win rate here.",
                "peak_win_rate": 0.70,
            },
            13: {
                # was level 6 - peak 42%, overall 7%.
                "platforms": [
                    pygame.Rect(60,530,150,20),
                    pygame.Rect(80,440,90,20),
                    pygame.Rect(220,370,80,20),
                    pygame.Rect(180,280,70,20),
                ],
                "player_start": (110,480),
                "star": pygame.Rect(200,220,30,30),
                "description": "Level 13 - Hard. Our best model achieved a 42% peak win rate here.",
                "peak_win_rate": 0.42,
            },
            14: {
                # was level 7 - peak 18%, overall 5%. 
                "platforms": [
                    pygame.Rect(40,520,150,20),
                    pygame.Rect(280,460,120,20),
                    pygame.Rect(80,390,100,20),
                    pygame.Rect(320,320,120,20),
                ],
                "player_start": (90,470),
                "star": pygame.Rect(360,260,30,30),
                "description": "Level 14 - Hard. Our best model achieved an 18% peak win rate here.",
                "peak_win_rate": 0.18,
            },
            15: {
                # was level 4 - 0% win rate across 726 attempts. 
                "platforms": [
                    pygame.Rect(50,550,150,20),
                    pygame.Rect(250,420,100,20),
                    pygame.Rect(100,250,120,20),
                    pygame.Rect(300,175,200,20),
                    pygame.Rect(360,350,30,20)
                ],
                "player_start": (70,500),
                "star": pygame.Rect(550,120,30,30),
                "description": "Level 15 - Hard. Our best model could not beat this level.",
                "peak_win_rate": 0.0,
            },
        }