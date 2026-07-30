# Floor Is Lava

## Overview

Floor Is Lava is a 2D platformer game and AI training simulator developed by Group 2 for the Summer 2026 semester of SDEV265.

In the game, a player navigates a series of platforms while avoiding falling into lava, with the goal of reaching a star. In addition to standard gameplay, the application includes an AI training mode, allowing users to observe and experiment with an AI agent learning to complete the level.

The application is runnable natively on Windows with a bundled version of the app in a `.exe` file, but there are alternaitve installation options for other operating systems.

# Installation

## Windows

For Windows users, the easiest way to run the application is through the bundled executable.

### Steps

1. Find the bundled executable in the `dist/` folder of this GitHub repository:

   https://github.com/nnguyen53/AI-Platformer/blob/main/dist/main.exe

2. Click **Download raw file** (or press **Ctrl + Shift + S**) to download `main.exe`.

3. Locate the downloaded file in your Downloads folder.

4. Double-click `main.exe`.

5. If Windows displays a security prompt, select **Run** to launch the application.

## Linux and macOS

If you are using a non-Windows operating system, you need to download and run the source code for the project to use the application.

### Prerequisites

- Python installed
- An IDE or code editor (such as VS Code)

### Setup

1. Navigate to the project repository:

   https://github.com/nnguyen53/AI-Platformer

2. Click **Code**, then **Download ZIP**.

3. Extract the ZIP file to your desired location.

4. Open the extracted folder in your IDE.

5. Create a virtual environment by running the following in your IDE terminal:

   ```bash
   python -m venv .venv
   ```

6. Activate the virtual environment.

   ```bash
   source .venv/bin/activate
   ```

7. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

8. Start the application:

   ```bash
   python main.py
   ```

## Running the Application

After launching the game, you can:

- Play the platformer manually using WAD + Space controls.
- Run the AI training mode to watch the AI learn and complete the level.
- Restart or rerun training as desired.
