import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import importlib.util
import sys
import os
from pathlib import Path
import threading
import tempfile
import ctypes

# Fix for high DPI displays (blurriness)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Dynamic Import of the Core Logic
# -----------------------------------------------------------------------------
script_path = Path(__file__).parent / "3d_to_2d_cad.py"
try:
    spec = importlib.util.spec_from_file_location("cad_converter", str(script_path))
    cad_module = importlib.util.module_from_spec(spec)
    sys.modules["cad_converter"] = cad_module
    spec.loader.exec_module(cad_module)
    Model3Dto2DConverter = cad_module.Model3Dto2DConverter
except Exception as e:
    # We will show this error in the GUI initialization if possible, or print it
    print(f"CRITICAL ERROR: Could not load 3d_to_2d_cad.py: {e}")
    Model3Dto2DConverter = None

# -----------------------------------------------------------------------------
# Main Application Class
# -----------------------------------------------------------------------------
class CADConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Professional 3D CAD to 2D Technical Drawing Converter")
        self.root.geometry("1000x800")
        
        # State
        self.current_file_path = None
        self.converter = None
        self.preview_image_path = None
        
        # Styles
        self.style = ttk.Style()
        self.style.theme_use('clam')  # 'clam' usually looks cleaner than default
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", padding=5)
        self.style.configure("TLabel", background="#f0f0f0")
        self.style.configure("TCheckbutton", background="#f0f0f0")
        
        self.root.configure(bg="#f0f0f0")

        self.create_layout()
        
        if Model3Dto2DConverter is None:
            messagebox.showerror("Error", "Could not load 3d_to_2d_cad.py. Ensure it is in the same directory.")

    def create_layout(self):
        # Top Control Panel
        control_frame = ttk.LabelFrame(self.root, text="Drawing Controls", padding=10)
        control_frame.pack(side="top", fill="x", padx=10, pady=5)
        
        # Row 1: File Loading
        row1 = ttk.Frame(control_frame)
        row1.pack(fill="x", pady=2)
        
        ttk.Button(row1, text="Load STL File", command=lambda: self.load_file(['stl'])).pack(side="left", padx=2)
        ttk.Button(row1, text="Load STEP File", command=lambda: self.load_file(['stp', 'step'])).pack(side="left", padx=2)
        ttk.Button(row1, text="Load OBJ File", command=lambda: self.load_file(['obj'])).pack(side="left", padx=2)
        
        self.file_label = ttk.Label(row1, text="No file selected", foreground="red")
        self.file_label.pack(side="left", padx=10)

        # Row 2: Actions
        row2 = ttk.Frame(control_frame)
        row2.pack(fill="x", pady=5)
        
        ttk.Button(row2, text="Generate Preview (PNG)", command=self.generate_preview).pack(side="left", padx=2)
        ttk.Button(row2, text="Export PNG", command=self.export_png).pack(side="left", padx=2)
        ttk.Button(row2, text="Export DXF", command=self.export_dxf).pack(side="left", padx=2)
        
        # Main Content Area (Split: Settings Left, Canvas Right)
        main_pane = ttk.PanedWindow(self.root, orient="horizontal")
        main_pane.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Left Sidebar: Settings
        settings_frame = ttk.Labelframe(main_pane, text="Drawing Settings", padding=10)
        main_pane.add(settings_frame, weight=1)
        
        # Views Selection
        ttk.Label(settings_frame, text="Views to Generate:", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(10, 5))
        
        # Standard views
        self.var_front = tk.BooleanVar(value=True)
        self.var_top = tk.BooleanVar(value=True)
        self.var_side = tk.BooleanVar(value=True) # Right/Side
        self.var_iso = tk.BooleanVar(value=False)
        
        # Section views (new)
        self.var_section_front = tk.BooleanVar(value=False)
        self.var_section_top = tk.BooleanVar(value=False)
        self.var_section_side = tk.BooleanVar(value=False)
        
        ttk.Label(settings_frame, text="Standard Views:", font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 2))
        ttk.Checkbutton(settings_frame, text="Front View", variable=self.var_front).pack(anchor="w", padx=(10, 0))
        ttk.Checkbutton(settings_frame, text="Top View", variable=self.var_top).pack(anchor="w", padx=(10, 0))
        ttk.Checkbutton(settings_frame, text="Right/Side View", variable=self.var_side).pack(anchor="w", padx=(10, 0))
        ttk.Checkbutton(settings_frame, text="Isometric View", variable=self.var_iso).pack(anchor="w", padx=(10, 0))
        
        ttk.Label(settings_frame, text="Section Views:", font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(10, 2))
        ttk.Label(settings_frame, text="(Shows internal features)", font=("Segoe UI", 7, "italic"), foreground="#666").pack(anchor="w", pady=(0, 2))
        ttk.Checkbutton(settings_frame, text="Section Front", variable=self.var_section_front).pack(anchor="w", padx=(10, 0))
        ttk.Checkbutton(settings_frame, text="Section Top", variable=self.var_section_top).pack(anchor="w", padx=(10, 0))
        ttk.Checkbutton(settings_frame, text="Section Side", variable=self.var_section_side).pack(anchor="w", padx=(10, 0))
        
        ttk.Separator(settings_frame, orient="horizontal").pack(fill="x", pady=15)
        
        ttk.Label(settings_frame, text="Note:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        ttk.Label(settings_frame, text="Line weights and hidden lines\nare auto-handled by the\nconverter script.", wraplength=150).pack(anchor="w")

        # Right Area: Canvas for Drawing Sheet
        sheet_frame = ttk.Labelframe(main_pane, text="Technical Drawing Sheet", padding=5)
        main_pane.add(sheet_frame, weight=4)
        
        self.canvas_container = ttk.Frame(sheet_frame)
        self.canvas_container.pack(fill="both", expand=True)
        
        # Scrollbars for canvas
        h_scroll = ttk.Scrollbar(self.canvas_container, orient="horizontal")
        v_scroll = ttk.Scrollbar(self.canvas_container, orient="vertical")
        
        self.canvas = tk.Canvas(self.canvas_container, bg="white", 
                                xscrollcommand=h_scroll.set, 
                                yscrollcommand=v_scroll.set)
        
        h_scroll.config(command=self.canvas.xview)
        v_scroll.config(command=self.canvas.yview)
        
        h_scroll.pack(side="bottom", fill="x")
        v_scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Bind resize event
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Placeholder text
        self.canvas_text = self.canvas.create_text(300, 200, text="Load a model and click 'Generate Preview'", fill="#aaa", font=("Arial", 14))

        # Bottom Status Bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
        
        self.original_image = None
        self.tk_img = None

    def on_canvas_configure(self, event):
        if self.original_image:
            self.display_fit_image()

    def load_file(self, file_types):
        ftypes = []
        if 'stl' in file_types: ftypes.append(("STL Files", "*.stl"))
        if 'stp' in file_types: ftypes.append(("STEP Files", "*.stp;*.step"))
        if 'obj' in file_types: ftypes.append(("OBJ Files", "*.obj"))
        ftypes.append(("All Files", "*.*"))
        
        path = filedialog.askopenfilename(filetypes=ftypes)
        if path:
            self.current_file_path = path
            self.file_label.config(text=os.path.basename(path), foreground="black")
            self.status_var.set(f"Loaded: {path}")
            
            # Reset converter
            try:
                self.converter = Model3Dto2DConverter(path)
                messagebox.showinfo("Success", f"Successfully loaded model:\n{os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
                self.converter = None

    def get_selected_views(self):
        views = []
        # Standard views
        if self.var_front.get(): views.append('front')
        if self.var_top.get(): views.append('top')
        if self.var_side.get(): views.append('side')
        if self.var_iso.get(): views.append('isometric')
        # Section views
        if self.var_section_front.get(): views.append('section_front')
        if self.var_section_top.get(): views.append('section_top')
        if self.var_section_side.get(): views.append('section_side')
        return views

    def generate_preview(self):
        if not self.converter:
            messagebox.showwarning("Warning", "Please load a file first.")
            return
            
        views = self.get_selected_views()
        if not views:
            messagebox.showwarning("Warning", "Select at least one view.")
            return
            
        def run_generation():
            self.status_var.set("Generating preview... please wait.")
            self.root.config(cursor="watch")
            
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                
                # Close the file so the converter can write to it
                
                self.converter.convert(output_format='png', output_path=tmp_path, views=views)
                
                # Update UI in main thread
                self.root.after(0, lambda: self.display_image(tmp_path))
                
            except Exception as e:
                self.root.after(0, lambda p=e: messagebox.showerror("Error", f"Generation failed: {p}"))
            finally:
                self.root.after(0, lambda: self.cleanup_ui_state())

        threading.Thread(target=run_generation, daemon=True).start()

    def cleanup_ui_state(self):
        self.status_var.set("Ready")
        self.root.config(cursor="")

    def display_image(self, img_path):
        try:
            self.preview_image_path = img_path
            # Load original image and keep reference
            self.original_image = Image.open(img_path)
            self.display_fit_image()
            self.status_var.set("Preview generated.")
            
        except Exception as e:
            messagebox.showerror("Display Error", f"Could not display image: {e}")

    def display_fit_image(self):
        if not self.original_image:
            return

        # Get canvas dimensions
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        if cw < 10 or ch < 10: 
            return # Metric not ready
        
        # Calculate scale to fit with padding
        iw, ih = self.original_image.size
        ratio = min(cw/iw, ch/ih) * 0.95 # 5% padding
        
        new_w = int(iw * ratio)
        new_h = int(ih * ratio)
        
        if new_w <= 0 or new_h <= 0:
            return

        # Resize
        try:
            resized = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        except AttributeError:
             # Fallback for older Pillow versions
            resized = self.original_image.resize((new_w, new_h), Image.ANTIALIAS)
            
        self.tk_img = ImageTk.PhotoImage(resized)
        
        self.canvas.delete("all")
        # Center image
        self.canvas.create_image(cw/2, ch/2, image=self.tk_img, anchor="center")

    def export_png(self):
        if not self.converter:
            messagebox.showwarning("Warning", "Please load a file first.")
            return
            
        views = self.get_selected_views()
        path = filedialog.asksaveasfilename(defaultextension=".png", 
                                          filetypes=[("PNG Image", "*.png")])
        if path:
            try:
                self.converter.convert(output_format='png', output_path=path, views=views)
                messagebox.showinfo("Success", f"Exported to {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")

    def export_dxf(self):
        if not self.converter:
            messagebox.showwarning("Warning", "Please load a file first.")
            return
            
        views = self.get_selected_views()
        path = filedialog.asksaveasfilename(defaultextension=".dxf", 
                                          filetypes=[("DXF Files", "*.dxf")])
        if path:
            try:
                self.converter.convert(output_format='dxf', output_path=path, views=views)
                messagebox.showinfo("Success", f"Exported to {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = CADConverterApp(root)
        root.mainloop()
    except Exception as e:
        # Fallback error reporting
        import traceback
        traceback.print_exc()
        input("Press Enter to close window...")
