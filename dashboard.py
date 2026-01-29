import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import sys
import os
import queue
from PIL import Image, ImageDraw
import pystray

# --- CONFIGURATION (Bot Definitions) ---
# Triple-V2 ARKK Deployment Strategy (The Proven Champion)
BOT_CONFIGS = [
    {
        "name": "Acc1: V6-ARKK (Uncapped)",
        "cmd": [
            "python", "deploy_bot.py", "--bot", "v2", "--symbol", "ARKK", 
            "--daily_target", "999", 
            "--key", "YOUR_API_KEY_HERE", 
            "--secret", "YOUR_SECRET_KEY_HERE", 
            "--url", "https://paper-api.alpaca.markets"
        ]
    },
    {
        "name": "Acc2: V2-ARKK (Uncapped)",
        "cmd": [
            "python", "deploy_bot.py", "--bot", "v2", "--symbol", "ARKK", 
            "--daily_target", "999", 
            "--key", "YOUR_API_KEY_HERE", 
            "--secret", "YOUR_SECRET_KEY_HERE", 
            "--url", "https://paper-api.alpaca.markets"
        ]
    },
    {
        "name": "Acc3: V2-ARKK (1% Cap)",
        "cmd": [
            "python", "deploy_bot.py", "--bot", "v2", "--symbol", "ARKK", 
            "--daily_target", "0.01", 
            "--key", "YOUR_API_KEY_HERE", 
            "--secret", "YOUR_SECRET_KEY_HERE", 
            "--url", "https://paper-api.alpaca.markets"
        ]
    }
]

class BotTab:
    def __init__(self, parent_notebook, config):
        self.config = config
        self.process = None
        self.is_running = False
        
        # Create Tab
        self.frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.frame, text=config["name"])
        
        # Controls
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.btn_start = ttk.Button(control_frame, text="Start", command=self.start_bot)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(control_frame, text="Stop", command=self.stop_bot, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="Status: STOPPED", foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Log Output (Create but don't pack yet)
        self.log_area = scrolledtext.ScrolledText(self.frame, state='disabled', height=10, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        
        # Log Queue (Thread-safe)
        self.log_queue = queue.Queue()
        self.frame.after(100, self.update_logs)
        
        # Parse symbol from cmd for the graph
        try:
            sym_idx = self.config["cmd"].index("--symbol") + 1
            symbol = self.config["cmd"][sym_idx]
        except:
            symbol = "UNKNOWN"
            
        # Layout: Simple Vertical Stacking
        # 1. Graph (Top)
        self.graph_panel = GraphPanel(self.frame, self.config['name'], symbol)
        self.graph_panel.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # 2. Logs (Bottom, fill remaining)
        self.log_area.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)

    def log(self, text):
        self.log_queue.put(text)

    def update_logs(self):
        while not self.log_queue.empty():
            text = self.log_queue.get()
            self.log_area.config(state='normal')
            self.log_area.insert(tk.END, text)
            self.log_area.see(tk.END)
            self.log_area.config(state='disabled')
        self.frame.after(100, self.update_logs)

    def start_bot(self):
        if self.is_running: return
        
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_label.config(text="Status: RUNNING", foreground="green")
        self.log(f"\n--- Starting {self.config['name']} ---\n")
        
        # Start subprocess thread
        threading.Thread(target=self._run_subprocess, daemon=True).start()

    def stop_bot(self):
        if not self.is_running or not self.process: return
        
        self.log("\n--- Stopping Bot... ---\n")
        # On Windows, we need to be forceful with taskkill because python.exe subprocesses can be sticky
        # But verify process.pid before killing
        if self.process:
             # Basic terminate
             self.process.terminate()
             # Wait a beat?
        
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_label.config(text="Status: STOPPED", foreground="red")

    def _run_subprocess(self):
        # Startup info to hide console window on Windows
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        # Force UTF-8 encoding to prevent crash on Emojis (Windows cp1252 default)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # Prepare command with unbuffered flag
        cmd = list(self.config["cmd"])
        if cmd[0] == "python" and "-u" not in cmd:
             cmd.insert(1, "-u")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW,
                env=env,
                encoding='utf-8' # Explicitly tell Python to read as UTF-8
            )
            
            for line in iter(self.process.stdout.readline, ''):
                self.log(line)
                
            self.process.stdout.close()
            return_code = self.process.wait()
            
            if self.is_running: # If it died unexpectedly
                self.log(f"\n[Process exited with code {return_code}]\n")
                self.stop_bot() # Update UI state
                
        except Exception as e:
            self.log(f"\n[Error starting process: {e}]\n")
            self.stop_bot()

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates

class GraphPanel(ttk.Frame):
    def __init__(self, parent, bot_name, symbol):
        super().__init__(parent)
        self.bot_name = bot_name
        self.symbol = symbol
        self.file_path = self._get_log_path()
        
        # Figure setup
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        self.fig.subplots_adjust(hspace=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Start update loop
        self.after(5000, self.update_graph) # Update every 5s

    def _get_log_path(self):
        # Reconstruct filename logic from deploy_bot.py
        # "logs/telemetry_{SYMBOL}_{safe_bot_name}.csv"
        # Note: bot_name comes from config['name'], which might change dynamically if it's an ensemble
        # But 'name' in config is "Acc 1: ARKK (Uncapped)" or similar.
        # deploy_bot uses model_data['name'].
        # Wait, deploy_bot uses `active_bot_name` which might be "Precision Sniper v2" OR "ENSEMBLE (Normal/Trend)".
        # The filename relies on `active_bot_name`. THIS IS TRICKY.
        # The dashboard doesn't know the internal state of the bot unless we parse the logs.
        # Workaround: Search for logs starting with telemetry_{SYMBOL}_ and pick the most recent modified?
        # Or simpler: Just look for *any* telemetry file for this symbol created recently.
        # Actually, let's just wildcard search in the update loop.
        return None 

    def update_graph(self):
        try:
            # Find the log file dynamically
            log_dir = "logs"
            if not os.path.exists(log_dir):
                self.after(5000, self.update_graph)
                return

            # Find candidates: telemetry_{symbol}_*.csv
            candidates = [f for f in os.listdir(log_dir) if f.startswith(f"telemetry_{self.symbol}")]
            if not candidates:
                self.after(5000, self.update_graph)
                return
                
            # Pick the most recently modified one
            candidates.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
            self.file_path = os.path.join(log_dir, candidates[0])
            
            # Read Data
            df = pd.read_csv(self.file_path)
            if df.empty:
                self.after(5000, self.update_graph)
                return
                
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            
            # Use last 100 points
            df = df.tail(100)
            
            # Plot 1: Price
            self.ax1.clear()
            self.ax1.plot(df.index, df['Price'], color='cyan', label='Price')
            
            # Buy Markers
            buys = df[df['Action'] == 1]
            if not buys.empty:
                self.ax1.scatter(buys.index, buys['Price'], color='lime', marker='^', s=50, label='Buy')
                
            # Sell Markers
            sells = df[df['Action'] == 2]
            if not sells.empty:
                self.ax1.scatter(sells.index, sells['Price'], color='red', marker='v', s=50, label='Sell')
            
            self.ax1.set_ylabel("Price")
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend(loc='upper left', fontsize='small')

            # Plot 2: Daily PnL
            self.ax2.clear()
            # Handle potential string format issues e.g. "0.01%" if captured as text, but we logged floats in deploy_bot
            self.ax2.fill_between(df.index, df['DailyPnL'], 0, where=(df['DailyPnL'] >= 0), color='green', alpha=0.3)
            self.ax2.fill_between(df.index, df['DailyPnL'], 0, where=(df['DailyPnL'] < 0), color='red', alpha=0.3)
            self.ax2.plot(df.index, df['DailyPnL'], color='white', linewidth=1)
            
            self.ax2.axhline(0, color='gray', linestyle='--')
            self.ax2.set_ylabel("Daily PnL")
            self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Graph error: {e}")
            
        self.after(5000, self.update_graph)

class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NASDAQ Bot Fleet Dashboard 🦅")
        self.root.geometry("1000x700")
        
        # Control Panel (Global) - Pack FIRST to ensure visibility at bottom
        btn_frame = ttk.Frame(root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="Start All", command=self.start_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Stop All", command=self.stop_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Quit Completely", command=self.quit_app).pack(side=tk.RIGHT, padx=5)

        # Notebook for Tabs - Pack SECOND to fill remaining space
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Create Tabs
        self.tabs = []
        for config in BOT_CONFIGS:
            tab = BotTab(self.notebook, config)
            self.tabs.append(tab)
            
        # System Tray logic
        self.root.protocol('WM_DELETE_WINDOW', self.hide_window)

    def start_all(self):
        for tab in self.tabs:
            tab.start_bot()
            
    def stop_all(self):
        for tab in self.tabs:
            tab.stop_bot()

    def hide_window(self):
        self.root.withdraw()
        if not hasattr(self, 'icon'):
            self.create_tray_icon()
        self.icon.visible = True
        
    def show_window(self):
        self.icon.visible = False
        self.root.deiconify()

    def create_icon_image(self):
        # Generate a simple icon image
        width = 64
        height = 64
        color1 = (0, 0, 0)
        color2 = (0, 255, 0)
        image = Image.new('RGB', (width, height), color1)
        dc = ImageDraw.Draw(image)
        dc.rectangle((16, 16, 48, 48), fill=color2)
        return image

    def create_tray_icon(self):
        image = self.create_icon_image()
        menu = pystray.Menu(
            pystray.MenuItem('Show Dashboard', self.show_window),
            pystray.MenuItem('Quit', self.quit_app)
        )
        self.icon = pystray.Icon("NASDAQ Bot", image, "NASDAQ Bot Fleet", menu)
        
        # Run icon in separate thread to not block TK
        threading.Thread(target=self.icon.run, daemon=True).start()

    def quit_app(self):
        self.stop_all()
        if hasattr(self, 'icon'):
            self.icon.stop()
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    # Style
    style = ttk.Style()
    style.theme_use('clam')
    root.mainloop()
