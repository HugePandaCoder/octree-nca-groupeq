import tkinter as tk
from tkinter import ttk  # Improved widgets

class ProfessionalApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set the title and size of the window
        self.title("Professional UI")
        self.geometry("800x600")

        # Set a professional color scheme
        self.bg_color = "#2C3E50"
        self.fg_color = "#ECF0F1"
        self.headline_bg = "#E74C3C"  # Added a background color for the headline

        # Set the background color
        self.configure(bg=self.bg_color)

        # Create and place the widgets
        self.create_widgets()

    def create_widgets(self):
        # Enhanced Font settings for the headline
        title_font = ("Calibri", 30, "bold italic")
        button_font = ("Calibri", 12)
        text_font = ("Calibri", 12)

        # Create a frame for the title with a background color and a border
        title_frame = tk.Frame(self, bg=self.headline_bg, bd=1, relief="solid")
        title_frame.pack(pady=20, fill="x")

        # Create an enhanced title label inside the frame
        title = tk.Label(title_frame, text="Professional UI App", font=title_font, bg=self.headline_bg, fg=self.fg_color)
        title.pack(pady=10)

        # Create a frame for dropdowns and buttons
        frame = tk.Frame(self, bg=self.bg_color)
        frame.pack(pady=20)

        # Create the first dropdown with Calibri font
        option_menu_1 = ttk.Combobox(frame, values=("Option 1", "Option 2", "Option 3"), font=button_font)
        option_menu_1.grid(row=0, column=0, padx=10)

        # Create the second dropdown with Calibri font
        option_menu_2 = ttk.Combobox(frame, values=("Option A", "Option B", "Option C"), font=button_font)
        option_menu_2.grid(row=0, column=1, padx=10)

        # Create a button with Calibri font
        button = tk.Button(frame, text="Press Me", command=self.on_button_press, bg="#3498DB", fg=self.fg_color, font=button_font)
        button.grid(row=0, column=2, padx=10)

        # Create the main view as a Text widget with Calibri font
        self.main_view = tk.Text(self, height=20, width=80, bg="#34495E", fg=self.fg_color, font=text_font)
        self.main_view.pack(pady=20)
        self.main_view.insert(tk.END, "Main View\n")

    def on_button_press(self):
        # Insert some text into the main view when the button is pressed
        self.main_view.insert(tk.END, "Button was pressed!\n")

# Create and run the application
app = ProfessionalApp()
app.mainloop()
