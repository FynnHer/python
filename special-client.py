import tkinter as tk
import socket
from PIL import Image, ImageTk
import pygame
from tkinter import messagebox
import cv2
import threading
import time
import os
from screeninfo import get_monitors
from ffpyplayer.player import MediaPlayer


# Color scheme
COLORS = {
    "background": "#2B303A",
    "board": "#92DCE5",
    "text": "#EEE5E9",
    "button": "#7C7C7C",
    "accent": "#D64933",
    "miss": "#FFFFFF",
    "hit": "#FFFF00"
}

# Asset paths - make sure these files exist in an 'assets' folder
ASSETS = {
    "cursor": "assets/cursor.png",
    "siren": "assets/siren.wav",
    "videos": {
        "hit": "assets/hit.mp4",
        "miss": "assets/miss.mp4",
        "win": "assets/win.mp4",
        "lose": "assets/lose.mp4"
    }
}

VIDEOLENGTHS = {
    "hit": 15,
    "miss": 10,
    "win": 54,
    "lose": 54
}

class DraggableShip(tk.Label):
    def __init__(self, parent, length, game_instance, **kwargs):
        super().__init__(parent, **kwargs)
        self.length = length
        self.game = game_instance
        self.bind('<Button-1>', self.start_drag)
        self.bind('<B1-Motion>', self.drag)
        self.bind('<ButtonRelease-1>', self.stop_drag)
        self._drag_data = {"x": 0, "y": 0}
        self._original_x = 0
        self._original_y = 0

    def start_drag(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y
        self.configure(bg=COLORS["accent"])

    def drag(self, event):
        deltax = event.x - self._drag_data["x"]
        deltay = event.y - self._drag_data["y"]
        x = self.winfo_x() + deltax
        y = self.winfo_y() + deltay
        self.place(x=x, y=y)

    def stop_drag(self, event):
        try:
            board_x = self.winfo_rootx() - self.game.board_frame.winfo_rootx()
            board_y = self.winfo_rooty() - self.game.board_frame.winfo_rooty()
            grid_x = board_x // 30
            grid_y = board_y // 30
            
            if 0 <= grid_x < 10 and 0 <= grid_y < 10:
                if self.game.place_ship(grid_y, grid_x, self.length):
                    self.destroy()
                else:
                    self.place(x=self._original_x, y=self._original_y)
            else:
                self.place(x=self._original_x, y=self._original_y)
        except Exception as e:
            print(f"Error in ship placement: {e}")
            self.place(x=self._original_x, y=self._original_y)

class BattleshipGame(tk.Tk):
    def __init__(self):
        super().__init__()
        self.player = None
        self.testcell = None
        self.playing = False
        self.current_video = None
        self.winlossstop = False
        self.cap = None
        self.field = [[" " for _ in range(10)] for _ in range(10)]
        self.title("Battleship")
        self.geometry("1200x800")
        self.configure(bg=COLORS["background"])
        
        # Initialize pygame mixer for sound effects
        pygame.mixer.init()
        
        # Load custom cursor
        try:
            cursor_image = Image.open(ASSETS["cursor"])
            cursor_image = cursor_image.resize((32, 32), Image.Resampling.LANCZOS)
            self.target_cursor = ImageTk.PhotoImage(cursor_image)
        except Exception as e:
            print(f"Could not load cursor image: {e}")
            self.target_cursor = None
        
        self.setup_ui()
        self.selected_cell = None
        self.game_active = False

    def setup_ui(self):
        self.main_container = tk.Frame(self, bg=COLORS["background"])
        self.main_container.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Left frame for game board
        self.left_frame = tk.Frame(self.main_container, bg=COLORS["background"])
        self.left_frame.pack(side=tk.LEFT, expand=True, fill="both")
        
        self.board_frame = tk.Frame(self.left_frame, bg=COLORS["background"])
        self.board_frame.pack(pady=20)
        
        # Create board cells
        self.cells = [[self.create_cell(i, j) for j in range(10)] for i in range(10)]
        
        # Ships frame
        self.ships_frame = tk.Frame(self.left_frame, bg=COLORS["background"])
        self.ships_frame.pack(pady=20)
        self.create_ship_buttons()
        
        # Right frame for controls
        self.right_frame = tk.Frame(self.main_container, bg=COLORS["background"])
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill="both", padx=20)
        
        # Username entry
        tk.Label(self.right_frame, text="Username", font=("Arial", 14), 
                bg=COLORS["background"], fg=COLORS["text"]).pack(pady=(0, 10))
        self.username_entry = tk.Entry(self.right_frame, font=("Arial", 12),
                                     bg=COLORS["button"], fg=COLORS["text"])
        self.username_entry.pack()
        
        # Start button
        self.start_button = tk.Button(self.right_frame, text="Find Game",
                                    font=("Arial", 14), bg=COLORS["accent"],
                                    fg=COLORS["text"], command=self.start_game)
        self.start_button.pack(pady=20)

    def create_cell(self, i, j):
        cell = tk.Label(self.board_frame, width=4, height=2,
                       bg=COLORS["board"], relief="solid", borderwidth=1)
        cell.grid(row=i, column=j, padx=1, pady=1)
        cell.bind('<Button-1>', lambda e, x=i, y=j: self.cell_clicked(x, y))
        return cell

    def create_ship_buttons(self):
        ships = [(1, "â¬¤"), (2, "â¬¤â¬¤"), (3, "â¬¤â¬¤â¬¤"), (4, "â¬¤â¬¤â¬¤â¬¤")]
        for length, symbol in ships:
            ship = DraggableShip(self.ships_frame, length, self,
                               text=symbol, font=("Arial", 14),
                               bg=COLORS["button"], fg=COLORS["text"],
                               padx=10, pady=5)
            ship.pack(side=tk.LEFT, padx=5)
            ship._original_x = ship.winfo_x()
            ship._original_y = ship.winfo_y()

    def place_ship(self, row, col, length):
        # Check if placement is valid
        if col + length > 10:
            return False
        if any(self.field[row][col + i] != " " for i in range(length)):
            return False
        
        # Place ship
        for i in range(length):
            self.field[row][col + i] = "o"
            self.cells[row][col + i].configure(bg=COLORS["accent"])
        
        # Enable start button if enough ships are placed
        if sum(row.count("o") for row in self.field) >= 10:
            self.start_button.configure(state="normal")
        return True

    def start_game(self):
        username = self.username_entry.get()
        if not username:
            messagebox.showerror("Error", "Please enter a username")
            return

        # Create socket connection
        self.komm_s = socket.socket()
        try:
            self.komm_s.connect(('127.0.0.1', 5000))
            self.send_str(username)
            
            response = self.receive_str()
            if response == "game start":
                self.game_active = True
                opponent_name = self.receive_str()
                self.send_str(str(self.field))
                self.setup_game_screen(opponent_name)
                
                # Start game loop in separate thread
                threading.Thread(target=self.game_loop, daemon=True).start()
            else:
                messagebox.showerror("Error", f"Unexpected response: {response}")
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))

    def setup_game_screen(self, opponent_name):
        # Clear window
        for widget in self.winfo_children():
            widget.destroy()
            
        # Create game container
        self.game_container = tk.Frame(self, bg=COLORS["background"])
        self.game_container.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Header with turn indicator
        self.header_frame = tk.Frame(self.game_container, bg=COLORS["background"])
        self.header_frame.pack(fill="x", pady=(0, 20))
        
        self.turn_label = tk.Label(self.header_frame, text="Waiting for opponent...",
                                 font=("Arial", 20, "bold"),
                                 bg=COLORS["background"], fg=COLORS["accent"])
        self.turn_label.pack()
        
        # Game boards
        self.boards_frame = tk.Frame(self.game_container, bg=COLORS["background"])
        self.boards_frame.pack(expand=True, fill="both")
        
        # Your board
        your_board_frame = tk.Frame(self.boards_frame, bg=COLORS["background"])
        your_board_frame.pack(side=tk.LEFT, expand=True, padx=20)
        tk.Label(your_board_frame, text="Your Fleet",
                font=("Arial", 16), bg=COLORS["background"],
                fg=COLORS["text"]).pack(pady=(0, 10))
        self.your_board = self.create_board(your_board_frame, is_opponent=False)
        
        # Opponent's board
        opponent_board_frame = tk.Frame(self.boards_frame, bg=COLORS["background"])
        opponent_board_frame.pack(side=tk.RIGHT, expand=True, padx=20)
        tk.Label(opponent_board_frame, text=f"{opponent_name}'s Fleet",
                font=("Arial", 16), bg=COLORS["background"],
                fg=COLORS["text"]).pack(pady=(0, 10))
        self.opponent_board = self.create_board(opponent_board_frame, is_opponent=True)
        
        # Launch button
        self.launch_button = tk.Button(self.game_container, text="LAUNCH",
                                     font=("Arial", 20, "bold"),
                                     bg=COLORS["accent"], fg=COLORS["text"],
                                     command=self.make_move,
                                     state="disabled",
                                     relief="raised",
                                     borderwidth=5,
                                     width=15, height=2)
        self.launch_button.pack(pady=20)

    def create_board(self, parent, is_opponent=False):
        board_frame = tk.Frame(parent, bg=COLORS["background"])
        board_frame.pack()
        
        cells = []
        for i in range(10):
            row = []
            for j in range(10):
                cell = tk.Label(board_frame, width=4, height=2,
                              bg=COLORS["board"], relief="solid", borderwidth=1)
                cell.grid(row=i, column=j, padx=1, pady=1)
                if is_opponent:
                    cell.bind('<Button-1>', lambda e, x=i, y=j: self.target_cell(x, y))
                row.append(cell)
            cells.append(row)
        return cells

    def target_cell(self, row, col):
        if not self.game_active or self.launch_button["state"] == "disabled":
            return
            
        # Reset previous selection
        if hasattr(self, 'selected_cell'):
            if self.selected_cell:
                old_row, old_col = self.selected_cell
                self.opponent_board[old_row][old_col].configure(
                    bg=COLORS["board"])
                
                # Reset cursor for all cells
                for r in self.opponent_board:
                    for cell in r:
                        cell.configure(cursor="")
        
        # Set new selection
        self.selected_cell = (row, col)
        self.opponent_board[row][col].configure(bg=COLORS["accent"])
        self.launch_button.configure(state="normal")
        
        # Change cursor for opponent's board cells when it's player's turn
        if self.launch_button["state"] != "disabled":
            for r in self.opponent_board:
                for cell in r:
                    if self.target_cursor:
                        cell.configure(cursor="crosshair")
                    else:
                        cell.configure(cursor="target")

    def play_alert(self):
        try:
            pygame.mixer.music.load(ASSETS["siren"])
            pygame.mixer.music.play()
            
            # Enhanced flash effect
            def flash():
                flash_colors = ["#0000FF", "#000080", "#0000FF", "#000080"]
                flash_durations = [0.3, 0.2, 0.3, 0.2]
                
                for color, duration in zip(flash_colors, flash_durations):
                    # Flash the main window
                    self.configure(bg=color)
                    
                    # Flash the board frames
                    for frame in [self.game_container, self.boards_frame]:
                        frame.configure(bg=color)
                    
                    self.update()
                    time.sleep(duration)
                    
                    # Reset colors
                    self.configure(bg=COLORS["background"])
                    for frame in [self.game_container, self.boards_frame]:
                        frame.configure(bg=COLORS["background"])
                    
                    self.update()
                    time.sleep(duration)
            
            threading.Thread(target=flash, daemon=True).start()
        except Exception as e:
            print(f"Error playing alert: {e}")

    def play_video(self, video_type):
        try:
            # Stop any existing playback first
            self.stop_current_playback()
            
            # Wait a brief moment to ensure cleanup is complete
            self.after(100)  # Add small delay for cleanup
            
            # Paths for video and duration in milliseconds
            video_path = ASSETS["videos"][video_type]
            duration = VIDEOLENGTHS[video_type] * 1000  # Duration in milliseconds

            # Store current video type
            self.current_video = video_type
                
            # Create a new fullscreen window for the video
            video_window = tk.Toplevel(self)
            video_window.attributes("-fullscreen", True)
            video_window.title(video_type.capitalize())
            
            # Create a label to display the video
            video_label = tk.Label(video_window)
            video_label.pack(expand=True, fill="both")

            # Initialize MediaPlayer and video capture
            self.player = MediaPlayer(video_path)
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                raise Exception(f"Failed to open video file: {video_path}")

            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_delay = int(1000 / fps)  # Convert fps to milliseconds delay
            
            self.playing = True

            def update_frame():
                if not self.playing or not self.cap or not self.cap.isOpened():
                    on_close()
                    return

                ret, frame = self.cap.read()
                if not ret:
                    on_close()
                    return

                # Resize frame to fit the screen while maintaining aspect ratio
                screen_width = video_window.winfo_screenwidth()
                screen_height = video_window.winfo_screenheight()
                
                # Calculate aspect ratio
                height, width = frame.shape[:2]
                aspect_ratio = width / height
                
                # Calculate new dimensions
                if screen_width / screen_height > aspect_ratio:
                    new_width = int(screen_height * aspect_ratio)
                    new_height = screen_height
                else:
                    new_width = screen_width
                    new_height = int(screen_width / aspect_ratio)
                
                frame = cv2.resize(frame, (new_width, new_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)

                # Only update if window still exists
                if video_window.winfo_exists():
                    video_label.configure(image=frame)
                    video_label.image = frame  # Keep a reference

                    if self.playing:
                        video_window.after(frame_delay, update_frame)

            def on_close():
                self.stop_current_playback()
                if video_window.winfo_exists():
                    video_window.destroy()

            # Handle window closing
            video_window.protocol("WM_DELETE_WINDOW", on_close)

            # Start playback
            if self.player:
                self.player.set_pause(False)
            update_frame()

            # Auto-close after duration
            video_window.after(duration, on_close)

        except Exception as e:
            print(f"Error playing video: {e}")
            self.stop_current_playback()
            if 'video_window' in locals() and video_window.winfo_exists():
                video_window.destroy()

    def stop_current_playback(self):
        """Safely stop current playback and cleanup resources."""
        self.playing = False
        
        # Cleanup MediaPlayer
        if self.player:
            try:
                self.player.set_pause(True)
                self.player.close_player()
                self.player = None
            except Exception as e:
                print(f"Error cleaning up player: {e}")

        # Cleanup Video Capture
        if self.cap:
            try:
                self.cap.release()
                self.cap = None
            except Exception as e:
                print(f"Error cleaning up video capture: {e}")
        
        self.current_video = None

    def stop_current_playback(self):
        """Safely stop current playback and cleanup resources."""
        self.playing = False
        if self.player:
            try:
                self.player.set_pause(True)
            except Exception:
                pass  # Ignore errors during cleanup
            self.player = None

    def game_loop(self):
        while self.game_active:
            try:
                print(self.testcell)
                message = self.receive_str()
                print(f"Received message: {message}")  # Debug print
                
                if message == "your turn":
                    self.turn_label.configure(text="Your Turn!")
                    self.launch_button.configure(state="normal")
                    self.play_alert()
                    for row in self.opponent_board:
                        for cell in row:
                            if self.target_cursor:
                                cell.configure(cursor="crosshair")
                            else:
                                cell.configure(cursor="target")
                
                elif message == "opponent turn":
                    self.turn_label.configure(text="Opponent's Turn...")
                    self.launch_button.configure(state="disabled")
                    for row in self.opponent_board:
                        for cell in row:
                            cell.configure(cursor="")

                elif message == "continue":
                    pass

                elif message == "winner":
                    self.game_active = False
                    self.play_video("win")
                    self.turn_label.configure(text="Victory!")
                    self.after(60000, self.reset_game)
                    self.winlossstop = False
                
                elif message == "looser":
                    self.game_active = False
                    self.play_video("lose")
                    self.turn_label.configure(text="Defeat!")
                    self.after(60000, self.reset_game)  

                # Handle the field updates
                elif isinstance(eval(message), dict):
                    fields = eval(message)
                    # Update own field
                    for i in range(10):
                        for j in range(10):
                            value = fields['own'][i][j]
                            if value == "s":
                                self.your_board[i][j].configure(bg=COLORS["hit"])
                            elif value == "x":
                                self.your_board[i][j].configure(bg=COLORS["miss"])
                            elif value == "o":
                                self.your_board[i][j].configure(bg=COLORS["accent"])
                    
                    # Update opponent field
                    for i in range(10):
                        for j in range(10):
                            value = fields['opponent'][i][j]
                            if value == "s":
                                self.opponent_board[i][j].configure(bg=COLORS["hit"])
                                if self.testcell == (i, j) and self.game_active:
                                    self.play_video("hit")
                            elif value == "x":
                                self.opponent_board[i][j].configure(bg=COLORS["miss"])
                                if self.testcell == (i, j) and self.game_active:
                                    self.play_video("miss")


                
                
            except Exception as e:
                print(f"Error in game loop: {e}")
                self.game_active = False
                break

    def make_move(self):
        if not hasattr(self, 'selected_cell') or not self.selected_cell:
            messagebox.showinfo("Select Target", "Please select a target cell first!")
            return
            
        row, col = self.selected_cell
        try:
            # Format move command to match server expectations
            # Server expects a tuple string like "(row,column)"
            move_cmd = str((row, col))
            self.send_str(move_cmd)
            
            # Disable launch button and reset selection
            self.launch_button.configure(state="disabled")
            self.opponent_board[row][col].configure(bg=COLORS["board"])
            
            # Reset cursor
            for r in self.opponent_board:
                for cell in r:
                    cell.configure(cursor="")
            
            self.testcell = self.selected_cell

            self.selected_cell = None
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send move: {e}")

    def reset_game(self):
        try:
            self.komm_s.close()
        except:
            pass
        self.destroy()
        BattleshipGame().mainloop()

    def send_str(self, data):
        try:
            self.komm_s.sendall(bytes(data, 'utf-8'))
            self.komm_s.sendall(bytes([0]))
        except Exception as e:
            raise Exception(f"Failed to send data: {e}")

    def receive_str(self):
        try:
            data_bytes = bytes()
            end_byte = bytes([0])
            
            while True:
                chunk = self.komm_s.recv(1)
                if chunk == end_byte or chunk == bytes([]):
                    break
                data_bytes += chunk
            
            return str(data_bytes, 'utf-8')
        except Exception as e:
            raise Exception(f"Failed to receive data: {e}")

    def __del__(self):
        try:
            self.komm_s.close()
        except:
            pass

if __name__ == "__main__":
    # Create assets directory if it doesn't exist
    if not os.path.exists('assets'):
        os.makedirs('assets')
        print("Created assets directory. Please add the following files:")
        print("- assets/cursor.png (cursor image)")
        print("- assets/siren.wav (alert sound)")
        print("- assets/hit.mp4 (hit animation)")
        print("- assets/miss.mp4 (miss animation)")
        print("- assets/victory.mp4 (win animation)")
        print("- assets/defeat.mp4 (lose animation)")
    
    game = BattleshipGame()
    game.mainloop()