import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
from pathlib import Path
import tkinter as tk

COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 200, 200),
    "green": (0, 200, 0),
    "orange": (0, 120, 255),
    "purple": (220, 0, 150)
}

class CTkGUILabeller:
    def __init__(self, board, predictions, tile_classes, max_crowns):
        self.board = board
        self.predictions = predictions
        self.tile_classes = tile_classes
        self.max_crowns = max_crowns
        
        # Determine if this is manual labeling mode (no real predictions)
        self.is_manual_mode = all(
            pred.get("tile_class_conf", 0) == 0 and pred.get("crown_count_conf", 0) == 0
            for pred in predictions.values()
        ) if predictions else True
        
        # Load settings
        settings_path = Path(__file__).parent.parent.parent / "settings.json"
        with open(settings_path, "r") as f:
            settings = json.load(f)
        self.board_width = settings["board"]["width"]
        self.board_height = settings["board"]["height"]
        
        self.selected_tiles = set()  # Changed to set for multi-selection
        self.last_selected_tile = None  # For shift-click range selection
        self.corrections = {}
        self.result = None
        self.current_preview_tile = None
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        mode_text = "Manual Labelling" if self.is_manual_mode else "Human-in-the-Loop Labelling"
        self.root.title(f"Board {board.index} - {mode_text}")
        
        # Handle window close button (X)
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        # Calculate window size
        board_height = self.board.image.shape[0]
        board_width = self.board.image.shape[1]
        window_width = board_width + 420
        window_height = board_height + 200  # Extra space for instructions below
        
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.resizable(True, True)
        
        self._create_ui()
        self._update_board_display()
        
    def _create_ui(self):
        """Create the modern UI layout"""
        # Main container
        self.root.grid_columnconfigure(0, weight=0)  # Board column (fixed)
        self.root.grid_columnconfigure(1, weight=1)  # Right panel (expandable)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        
        # Left side - Board and instructions
        left_container = ctk.CTkFrame(self.root, fg_color="transparent")
        left_container.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(10, 5), pady=10)
        
        # Canvas for board
        self.canvas = tk.Canvas(
            left_container,
            width=self.board.image.shape[1],
            height=self.board.image.shape[0],
            bg="#1a1a1a",
            highlightthickness=0
        )
        self.canvas.pack(pady=(0, 10))
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        
        # Instructions below board
        instructions_frame = ctk.CTkFrame(left_container, corner_radius=10)
        instructions_frame.pack(fill="x")
        
        ctk.CTkLabel(
            instructions_frame,
            text="📋 Instructions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=15, pady=(10, 5))
        
        if self.is_manual_mode:
            instructions_text = (
                "> Click a tile to select/deselect\n"
                "> Hold Ctrl to select multiple tiles\n"
                "> Hold Shift to select a range\n"
                "> Colors:\n"
                ">> Orange = Selected\n"
                ">> Purple = Labeled\n"
                ">> Gray   = Not yet labeled"
            )
        else:
            instructions_text = (
                "> Click a tile to select/deselect\n"
                "> Hold Ctrl to select multiple tiles\n"
                "> Hold Shift to select a range\n"
                "> Colors:\n"
                ">> Orange = Selected\n"
                ">> Purple = Corrected\n"
                ">> Green  = High confidence   (>80%)\n"
                ">> Yellow = Medium confidence (50-80%)\n"
                ">> Red    = Low confidence    (<50%)"
            )
        ctk.CTkLabel(
            instructions_frame,
            text=instructions_text,
            font=ctk.CTkFont(size=11),
            justify="left"
        ).pack(anchor="w", padx=15, pady=(0, 10))
        
        # Right side - Control panel (non-scrollable to prevent button cutoff)
        right_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        right_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(5, 10), pady=10)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Selected tiles
        selected_frame = ctk.CTkFrame(right_frame, corner_radius=10)
        selected_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        selected_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            selected_frame,
            text="🎯 Selected Tiles",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 5))
        
        self.selected_label = ctk.CTkLabel(
            selected_frame,
            text="No tile selected",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#ffa500"
        )
        self.selected_label.grid(row=1, column=0, sticky="w", padx=15, pady=(0, 15))
        
        # Tile preview
        preview_frame = ctk.CTkFrame(right_frame, corner_radius=10)
        preview_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        preview_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            preview_frame,
            text="Tile Preview",
            font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(12, 8))
        
        # Tile title (above preview) - centered and bold
        self.preview_title = ctk.CTkLabel(
            preview_frame,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            justify="center"
        )
        self.preview_title.grid(row=1, column=0, padx=15, pady=(0, 4))
        
        # Preview label - 75% of original size (210x210 instead of 280x280)
        self.preview_label = ctk.CTkLabel(
            preview_frame,
            text="Click a tile",
            font=ctk.CTkFont(size=12),
            width=210,
            height=210,
            corner_radius=8,
            fg_color="#2b2b2b"
        )
        self.preview_label.grid(row=2, column=0, padx=15, pady=(0, 4))
        
        # Info below preview - centered
        self.preview_info = ctk.CTkLabel(
            preview_frame,
            text="",
            font=ctk.CTkFont(size=11),
            justify="center"
        )
        self.preview_info.grid(row=3, column=0, padx=15, pady=(0, 12))
        
        # Correction form
        correction_frame = ctk.CTkFrame(right_frame, corner_radius=10)
        correction_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        correction_frame.grid_columnconfigure(1, weight=1)
        
        correction_title = "📝 Label Selected Tiles" if self.is_manual_mode else "✏️ Correct Selected Tiles"
        ctk.CTkLabel(
            correction_frame,
            text=correction_title,
            font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(12, 8))
        
        # Tile class dropdown
        ctk.CTkLabel(
            correction_frame,
            text="Tile Class:",
            font=ctk.CTkFont(size=12)
        ).grid(row=1, column=0, sticky="w", padx=15, pady=8)
        
        self.class_var = tk.StringVar()
        self.class_combo = ctk.CTkComboBox(
            correction_frame,
            variable=self.class_var,
            values=[f"{i}: {cls}" for i, cls in enumerate(self.tile_classes)],
            width=220,
            font=ctk.CTkFont(size=12),
            state="readonly"
        )
        self.class_combo.grid(row=1, column=1, sticky="ew", padx=(5, 15), pady=8)
        
        # Crown count with slider
        ctk.CTkLabel(
            correction_frame,
            text="Crown Count:",
            font=ctk.CTkFont(size=12)
        ).grid(row=2, column=0, sticky="w", padx=15, pady=8)
        
        crown_container = ctk.CTkFrame(correction_frame, fg_color="transparent")
        crown_container.grid(row=2, column=1, sticky="ew", padx=(5, 15), pady=8)
        crown_container.grid_columnconfigure(1, weight=1)
        
        self.crown_var = tk.IntVar(value=0)
        self.crown_value_label = ctk.CTkLabel(
            crown_container,
            text="0",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=30
        )
        self.crown_value_label.grid(row=0, column=0, padx=(0, 10))
        
        self.crown_slider = ctk.CTkSlider(
            crown_container,
            from_=0,
            to=self.max_crowns,
            number_of_steps=self.max_crowns,
            variable=self.crown_var,
            command=self._update_crown_label,
            width=180
        )
        self.crown_slider.grid(row=0, column=1, sticky="ew")
        self.crown_slider.set(0)
        
        # Apply button
        apply_text = "✓ Apply Label" if self.is_manual_mode else "✓ Apply Correction"
        ctk.CTkButton(
            correction_frame,
            text=apply_text,
            command=self._apply_correction,
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            corner_radius=8,
            fg_color="#0078d4",
            hover_color="#106ebe"
        ).grid(row=3, column=0, columnspan=2, sticky="ew", padx=15, pady=(8, 12))
        
        # Action buttons
        buttons_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        buttons_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        buttons_frame.grid_columnconfigure(0, weight=1)
        
        accept_text = "✓ Accept All Labels" if self.is_manual_mode else "✓ Accept All Predictions"
        ctk.CTkButton(
            buttons_frame,
            text=accept_text,
            command=self._accept_all,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=42,
            corner_radius=8,
            fg_color="#4ec9b0",
            hover_color="#36e73f"
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        
        ctk.CTkButton(
            buttons_frame,
            text="⟲ Clear Selection",
            command=self._clear_selection,
            font=ctk.CTkFont(size=12),
            height=36,
            corner_radius=8,
            fg_color="#3e3e3e",
            hover_color="#4e4e4e"
        ).grid(row=1, column=0, sticky="ew")
    
    def _update_crown_label(self, value):
        """Update crown count label when slider moves"""
        crown_val = int(float(value))
        self.crown_value_label.configure(text=str(crown_val))
        self.crown_var.set(crown_val)
    
    def _create_board_image(self):
        """Create board visualization with predictions"""
        vis_image = self.board.image.copy()
        tile_height = vis_image.shape[0] // self.board_height
        tile_width = vis_image.shape[1] // self.board_width
        
        for tile_idx, pred in self.predictions.items():
            col = pred["col"]
            row = pred["row"]
            
            x1 = col * tile_width
            y1 = row * tile_height
            x2 = (col + 1) * tile_width
            y2 = (row + 1) * tile_height
            
            # Determine border color and thickness based on state
            border_thickness = 2
            inset = 4  # Consistent outline inset for all tiles
            
            if tile_idx in self.selected_tiles:
                color = COLORS["orange"] # Selected tiles
            elif tile_idx in self.corrections:
                color = COLORS["purple"] # Corrected tiles
            else:
                # Color based on confidence (or gray in manual mode)
                if self.is_manual_mode:
                    color = (128, 128, 128)  # Gray for unlabeled in manual mode
                else:
                    conf = min(pred["tile_class_conf"], pred["crown_count_conf"])
                    if conf >= 0.9:
                        color = COLORS["green"]  # High confidence
                    elif conf >= 0.7:
                        color = COLORS["yellow"]  # Medium confidence
                    else:
                        color = COLORS["red"]  # Low confidence

            # Draw rectangle with inset to keep border fully inside tile
            cv2.rectangle(vis_image, (x1 + inset, y1 + inset), (x2 - inset, y2 - inset), color, border_thickness)
            
            # Get current class/crown
            if tile_idx in self.corrections:
                tile_class = self.corrections[tile_idx]["tile_class"]
                crown_count = self.corrections[tile_idx]["crown_count"]
            else:
                tile_class = pred["tile_class"]
                crown_count = pred["crown_count"]
            
            # Force crown count to 0 for None (8), Crown (6), and Castle (7) tiles
            if tile_class in [6, 7, 8]:
                crown_count = 0
            
            # Add centered text with better rendering
            class_name = self.tile_classes[tile_class]  # Full name
            crown_text = f"C{crown_count}" if crown_count > 0 else ""
            
            # Calculate center position
            tile_center_x = (x1 + x2) // 2
            tile_center_y = (y1 + y2) // 2
            
            # Use bold font for better readability
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.45
            text_thickness = 1
            outline_thickness = 3
            
            # Get text sizes for centering
            (text_width1, text_height1), baseline1 = cv2.getTextSize(class_name, font, font_scale, text_thickness)
            (text_width2, text_height2), baseline2 = cv2.getTextSize(crown_text, font, font_scale, text_thickness)
            
            # Calculate text positions (centered)
            text1_x = tile_center_x - text_width1 // 2
            text1_y = tile_center_y - 8  # Above center
            text2_x = tile_center_x - text_width2 // 2
            text2_y = tile_center_y + text_height2 + 8  # Below center
            
            # Draw text with outline for better visibility
            outline_color = (0, 0, 0)
            text_color = (255, 255, 255)
            
            # Class name with outline
            cv2.putText(vis_image, class_name, (text1_x, text1_y), 
                       font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
            cv2.putText(vis_image, class_name, (text1_x, text1_y), 
                       font, font_scale, text_color, text_thickness, cv2.LINE_AA)
            
            # Crown count (if any) with outline
            if crown_text:
                cv2.putText(vis_image, crown_text, (text2_x, text2_y), 
                           font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
                cv2.putText(vis_image, crown_text, (text2_x, text2_y), 
                           font, font_scale, text_color, text_thickness, cv2.LINE_AA)
        
        return vis_image
    
    def _update_board_display(self):
        """Update the board canvas"""
        vis_image = self._create_board_image()
        
        # Convert BGR to RGB
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(vis_image_rgb)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.display_scale = 1.0
    
    def _on_canvas_click(self, event):
        """Handle canvas click with multi-selection support (Ctrl/Shift)"""
        x = event.x
        y = event.y
        
        tile_height = self.board.image.shape[0] // self.board_height
        tile_width = self.board.image.shape[1] // self.board_width
        
        col = x // tile_width
        row = y // tile_height
        
        if 0 <= col < self.board_width and 0 <= row < self.board_height:
            for tile_idx, pred in self.predictions.items():
                if pred["col"] == col and pred["row"] == row:
                    # Check for modifier keys
                    ctrl_pressed = (event.state & 0x4) != 0  # Ctrl key
                    shift_pressed = (event.state & 0x1) != 0  # Shift key
                    
                    if shift_pressed and self.last_selected_tile is not None:
                        # Shift-click: Range selection
                        self._select_range(self.last_selected_tile, tile_idx)
                    elif ctrl_pressed:
                        # Ctrl-click: Toggle individual tile
                        if tile_idx in self.selected_tiles:
                            self.selected_tiles.remove(tile_idx)
                        else:
                            self.selected_tiles.add(tile_idx)
                            self.last_selected_tile = tile_idx
                    else:
                        # Normal click: Single selection
                        if len(self.selected_tiles) == 1 and tile_idx in self.selected_tiles:
                            # Deselect if clicking the only selected tile
                            self.selected_tiles.clear()
                            self.last_selected_tile = None
                        else:
                            # Select only this tile
                            self.selected_tiles = {tile_idx}
                            self.last_selected_tile = tile_idx
                    
                    self._update_board_display()
                    self._update_selected_info()
                    
                    # Update preview based on selection
                    if len(self.selected_tiles) == 0:
                        self._clear_tile_preview()
                    elif len(self.selected_tiles) == 1:
                        self._show_tile_preview(tile_idx)
                    else:
                        self._show_multi_selection_preview()
                    break
    
    def _select_range(self, start_idx, end_idx):
        """Select all tiles in range from start to end (for Shift-click)"""
        # Get start and end positions
        start_pred = self.predictions[start_idx]
        end_pred = self.predictions[end_idx]
        
        start_pos = start_pred["row"] * self.board_width + start_pred["col"]
        end_pos = end_pred["row"] * self.board_width + end_pred["col"]
        
        # Ensure start is before end
        if start_pos > end_pos:
            start_pos, end_pos = end_pos, start_pos
        
        # Select all tiles in range
        for tile_idx, pred in self.predictions.items():
            tile_pos = pred["row"] * self.board_width + pred["col"]
            if start_pos <= tile_pos <= end_pos:
                self.selected_tiles.add(tile_idx)
        
        self.last_selected_tile = end_idx
    
    def _update_selected_info(self):
        """Update selected tiles info"""
        if len(self.selected_tiles) == 0:
            self.selected_label.configure(text="No tiles selected")
        elif len(self.selected_tiles) == 1:
            tile_idx = next(iter(self.selected_tiles))
            self.selected_label.configure(text=f"Selected: Tile {tile_idx}")
        else:
            tiles_str = ", ".join(str(t) for t in sorted(self.selected_tiles))
            if len(tiles_str) > 50:
                tiles_str = tiles_str[:47] + "..."
            self.selected_label.configure(text=f"Selected: {len(self.selected_tiles)} tiles ({tiles_str})")
    
    def _show_tile_preview(self, tile_idx):
        """Show preview of clicked tile"""
        tile = self.board.get_tile_from_index(tile_idx)
        if tile is None or tile.image is None or tile.image.size == 0:
            return
        
        self.current_preview_tile = tile_idx
        pred = self.predictions[tile_idx]
        
        # Convert tile image to RGB
        tile_rgb = cv2.cvtColor(tile.image, cv2.COLOR_BGR2RGB)
        pil_tile = Image.fromarray(tile_rgb)
        
        # Resize for preview - 75% of original (210x210 instead of 280x280)
        pil_tile = pil_tile.resize((210, 210), Image.Resampling.LANCZOS)
        
        # Convert to CTkImage
        ctk_image = ctk.CTkImage(light_image=pil_tile, dark_image=pil_tile, size=(210, 210))
        
        # Store reference to prevent garbage collection
        self.preview_image_ref = ctk_image
        self.preview_label.configure(image=ctk_image, text="")
        
        # Show info
        if tile_idx in self.corrections:
            tile_class = self.corrections[tile_idx]["tile_class"]
            crown_count = self.corrections[tile_idx]["crown_count"]
            title_text = f"Tile {tile_idx} (LABELED)" if self.is_manual_mode else f"Tile {tile_idx} (CORRECTED)"
            info_text = f"{self.tile_classes[tile_class]}\n{crown_count} crowns"
        else:
            tile_class = pred["tile_class"]
            crown_count = pred["crown_count"]
            
            if self.is_manual_mode:
                title_text = f"Tile {tile_idx}"
                info_text = "Not yet labeled"
            else:
                title_text = f"Tile {tile_idx} (Predicted)"
                info_text = (f"{self.tile_classes[tile_class]} ({pred['tile_class_conf']:.1%})\n"
                            f"{crown_count} crowns ({pred['crown_count_conf']:.1%})")
        
        self.preview_title.configure(text=title_text)
        self.preview_info.configure(text=info_text)
        
        # Pre-fill correction form
        self.class_combo.set(f"{tile_class}: {self.tile_classes[tile_class]}")
        self.crown_slider.set(crown_count)
        self.crown_value_label.configure(text=str(crown_count))
    
    def _show_multi_selection_preview(self):
        """Show preview for multiple selected tiles"""
        # Create a blank image with text
        blank_image = Image.new("RGB", (210, 210), color="#2b2b2b")
        blank_ctk_image = ctk.CTkImage(light_image=blank_image, dark_image=blank_image, size=(210, 210))
        
        # Set blank image with text overlay
        self.preview_image_ref = blank_ctk_image
        self.preview_label.configure(image=blank_ctk_image, text=f"{len(self.selected_tiles)} tiles\nselected")
        
        # Show title and info
        self.preview_title.configure(text=f"Multiple Tiles Selected ({len(self.selected_tiles)})")
        
        # Check if all selected tiles have the same class and crown count
        selected_classes = set()
        selected_crowns = set()
        for tile_idx in self.selected_tiles:
            if tile_idx in self.corrections:
                selected_classes.add(self.corrections[tile_idx]["tile_class"])
                selected_crowns.add(self.corrections[tile_idx]["crown_count"])
            else:
                selected_classes.add(self.predictions[tile_idx]["tile_class"])
                selected_crowns.add(self.predictions[tile_idx]["crown_count"])
        
        if len(selected_classes) == 1:
            # All same class - show it and allow crown editing
            tile_class = next(iter(selected_classes))
            self.preview_info.configure(
                text=f"All tiles are: {self.tile_classes[tile_class]}\n"
                     f"You can update crown count"
            )
            # Pre-fill the class dropdown
            self.class_combo.set(f"{tile_class}: {self.tile_classes[tile_class]}")
        else:
            # Different classes - only allow crown editing
            self.preview_info.configure(
                text=f"{len(selected_classes)} different classes\n"
                     f"Set crown count only (preserves classes)"
            )
            # Don't pre-fill class for mixed selection
            self.class_combo.set("")
        
        # Pre-fill crown count if all selected tiles have the same crown count
        if len(selected_crowns) == 1:
            crown_count = next(iter(selected_crowns))
            self.crown_slider.set(crown_count)
            self.crown_value_label.configure(text=str(crown_count))
        else:
            # Different crown counts - reset to 0
            self.crown_slider.set(0)
            self.crown_value_label.configure(text="0")
    
    def _clear_tile_preview(self):
        """Clear the tile preview when no tile is selected"""
        self.current_preview_tile = None
        
        # Create a blank image to avoid TclError
        blank_image = Image.new("RGB", (210, 210), color="#2b2b2b")
        blank_ctk_image = ctk.CTkImage(light_image=blank_image, dark_image=blank_image, size=(210, 210))
        
        # Set blank image and text
        self.preview_image_ref = blank_ctk_image
        self.preview_label.configure(image=blank_ctk_image, text="Click a tile")
        
        # Clear the title and info text
        self.preview_title.configure(text="")
        self.preview_info.configure(text="")
        
        # Reset correction form to defaults
        self.class_combo.set("")
        self.crown_slider.set(0)
        self.crown_value_label.configure(text="0")
    
    def _apply_correction(self):
        """Apply correction to selected tiles"""
        if len(self.selected_tiles) == 0:
            return
        
        class_str = self.class_var.get()
        crown_count = self.crown_var.get()
        
        # Check if we're only updating crown count (no class selected)
        crown_only = not class_str or class_str == ""
        
        if crown_only:
            # Update only crown counts, preserve existing classes
            action_text = "Updated crown count for" if not self.is_manual_mode else "Set crown count for"
            for tile_idx in self.selected_tiles:
                pred = self.predictions[tile_idx]
                
                # Get existing class (from corrections or predictions)
                if tile_idx in self.corrections:
                    existing_class = self.corrections[tile_idx]["tile_class"]
                else:
                    existing_class = pred["tile_class"]
                
                # Force crown count to 0 for None (8), Crown (6), and Castle (7) tiles
                actual_crown_count = crown_count
                if existing_class in [6, 7, 8]:
                    actual_crown_count = 0
                
                self.corrections[tile_idx] = {
                    "tile_class": existing_class,
                    "crown_count": actual_crown_count,
                    "col": pred["col"],
                    "row": pred["row"]
                }
            
            tiles_text = f"{len(self.selected_tiles)} tiles" if len(self.selected_tiles) > 1 else f"tile {next(iter(self.selected_tiles))}"
            print(f"✓ {action_text} {tiles_text} to: {crown_count} crown(s)")
        else:
            # Update both class and crown count
            tile_class = int(class_str.split(":")[0])
            
            # Force crown count to 0 for None (8), Crown (6), and Castle (7) tiles
            if tile_class in [6, 7, 8]:
                crown_count = 0
                print(f"ℹ Note: {self.tile_classes[tile_class]} tiles always have 0 crowns")
            
            # Apply to all selected tiles
            action_text = "Labeled" if self.is_manual_mode else "Corrected"
            for tile_idx in self.selected_tiles:
                pred = self.predictions[tile_idx]
                self.corrections[tile_idx] = {
                    "tile_class": tile_class,
                    "crown_count": crown_count,
                    "col": pred["col"],
                    "row": pred["row"]
                }
            
            tiles_text = f"{len(self.selected_tiles)} tiles" if len(self.selected_tiles) > 1 else f"tile {next(iter(self.selected_tiles))}"
            print(f"✓ {action_text} {tiles_text} to: {self.tile_classes[tile_class]}, {crown_count} crown(s)")
        
        self.selected_tiles.clear()
        self.last_selected_tile = None
        self._update_board_display()
        self._update_selected_info()
        self._clear_tile_preview()
    
    def _clear_selection(self):
        """Clear tile selection"""
        self.selected_tiles.clear()
        self.last_selected_tile = None
        self._update_board_display()
        self._update_selected_info()
        self._clear_tile_preview()
    
    def _accept_all(self):
        """Accept all predictions and close"""
        final_labels = {}
        for tile_idx, pred in self.predictions.items():
            if tile_idx in self.corrections:
                final_labels[str(tile_idx)] = self.corrections[tile_idx]
            else:
                final_labels[str(tile_idx)] = {
                    "tile_class": pred["tile_class"],
                    "crown_count": pred["crown_count"],
                    "col": pred["col"],
                    "row": pred["row"]
                }
        
        self.result = final_labels
        self._cleanup_and_close()
    
    def _cleanup_and_close(self):
        """Properly cleanup resources before closing"""
        try:
            # Cancel any pending after callbacks
            for after_id in self.root.tk.call('after', 'info'):
                self.root.after_cancel(after_id)
        except:
            pass  # Ignore errors during cleanup
        
        self.root.quit()
        self.root.destroy()
    
    def _on_window_close(self):
        """Handle window close button (X) - prompt user to confirm exit"""
        from tkinter import messagebox
        
        if messagebox.askyesno("Exit", "Do you want to halt execution and exit?\n\nThis will stop the labeling process."):
            print("\n⚠️ User halted execution via window close")
            self.result = None
            self._cleanup_and_close()
            import sys
            sys.exit(0)
    
    def run(self):
        """Run the GUI and return result"""
        self.root.mainloop()
        return self.result
