import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path

from components.board import Board
from multitask_classifier import MultiTaskTileClassifier
from train.ctk_gui_labeller import CTkGUILabeller

# Load settings
_settings_path = Path(__file__).parent.parent.parent / "settings.json"
with open(_settings_path, "r") as f:
    _SETTINGS = json.load(f)

class TileDataset(Dataset):
    def __init__(self, tiles_data, transform=None):
        self.tiles_data = tiles_data
        self.transform = transform
    
    def __len__(self):
        return len(self.tiles_data)
    
    def __getitem__(self, idx):
        tile_data = self.tiles_data[idx]
        image = Image.fromarray(tile_data["image"][:, :, ::-1])  # BGR to RGB
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "tile_class": tile_data["tile_class"],
            "crown_count": tile_data["crown_count"],
            "board_idx": tile_data["board_idx"],
            "tile_idx": tile_data["tile_idx"]
        }

class AutoLabeller:
    def __init__(self, save_dir=None, model_dir=None, labels_file=None, classes_file=None):
        # Load from settings if not provided
        self.model_dir = Path(model_dir) if model_dir else Path(_SETTINGS["paths"]["model_dir"])
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Load labels file
        self.labels_file = Path(labels_file) if labels_file else Path(_SETTINGS["paths"]["labels_file"])
        self.labels = self._load_labels()
        
        # Load tile classes from file
        classes_path = classes_file if classes_file else _SETTINGS["paths"]["classes_file"]
        self.tile_classes = self._load_classes(classes_path)
        self.num_classes = len(self.tile_classes)
        self.max_crowns = _SETTINGS["model"]["max_crowns"]
        
        self.board_width = _SETTINGS["board"]["width"]
        self.board_height = _SETTINGS["board"]["height"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Use settings for image size, or default to 224 for EfficientNet
        image_size = _SETTINGS["model"].get("image_size", 224)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def _load_labels(self):
        """Load existing labels from JSON file"""
        if self.labels_file.exists():
            with open(self.labels_file, "r") as f:
                return json.load(f)
        return {}
    
    def _load_classes(self, classes_file):
        """Load tile classes from classes.json file"""
        classes_path = Path(classes_file)
        if not classes_path.exists():
            print(f"Warning: {classes_file} not found, using default classes")
            return ["Meadow", "Woods", "Farmland", "Desert", "Mine", "Ocean", "Crown", "Castle", "None"]
        
        with open(classes_path, "r") as f:
            data = json.load(f)
            classes_data = data.get("classes", [])
            
            # Handle both list format and dict format
            if isinstance(classes_data, dict):
                # Convert dict to list, sorted by key
                classes = [classes_data[str(i)] for i in range(len(classes_data))]
            else:
                classes = classes_data
        
        print(f"Loaded {len(classes)} classes from {classes_file}")
        return classes
    
    def _save_labels(self):
        """Save labels to JSON file"""
        with open(self.labels_file, "w") as f:
            json.dump(self.labels, f, indent=2)
    
    def get_manually_labeled_boards(self):
        """Get list of boards that have been manually labeled"""
        return sorted([int(board_idx) for board_idx in self.labels.keys()])
    
    def manual_label_board(self, board_idx):
        """Manually label all tiles in a board"""
        print(f"\n=== Labeling Board {board_idx} ===")
        board = Board(board_idx)
        
        board_labels = {}
        
        for tile in board.tiles:
            # Get tile class
            print(f"\nTile {tile.index} - Position ({tile.col}, {tile.row})")
            print("Available classes:")
            for i, cls in enumerate(self.tile_classes):
                print(f"  {i}: {cls}")
            
            while True:
                try:
                    class_input = input("Enter tile class number: ")
                    tile_class = int(class_input)
                    if 0 <= tile_class < self.num_classes:
                        break
                    print(f"Invalid class. Please enter 0-{self.num_classes-1}")
                except ValueError:
                    print("Please enter a valid number")
            
            # Get crown count
            while True:
                try:
                    crown_input = input(f"Enter crown count (0-{self.max_crowns}): ")
                    crown_count = int(crown_input)
                    if 0 <= crown_count <= self.max_crowns:
                        break
                    print(f"Invalid count. Please enter 0-{self.max_crowns}")
                except ValueError:
                    print("Please enter a valid number")
            
            # Store label
            board_labels[str(tile.index)] = {
                "tile_class": tile_class,
                "crown_count": crown_count,
                "col": tile.col,
                "row": tile.row
            }
            
            print(f"✓ Labeled as: {self.tile_classes[tile_class]}, {crown_count} crown(s)")
        
        self.labels[str(board_idx)] = board_labels
        self._save_labels()
        print(f"\n✓ Board {board_idx} fully labeled and saved!")
    
    def train_model(self, num_epochs=None, batch_size=None, training_boards_num=None):
        """Train model on manually labeled boards
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            training_boards_num: Optional - randomly sample this many boards for training (for debugging)
        """
        if num_epochs is None:
            num_epochs = _SETTINGS["training"]["initial_epochs"]
        if batch_size is None:
            batch_size = _SETTINGS["training"]["batch_size"]
        
        print("\n=== Training Model ===")
        
        # Optionally sample a subset of boards for faster training (debug mode)
        boards_to_train = list(self.labels.keys())
        if training_boards_num is not None and training_boards_num < len(boards_to_train):
            import random
            boards_to_train = random.sample(boards_to_train, training_boards_num)
            print(f"🔧 DEBUG MODE: Using {training_boards_num} random boards: {sorted([int(b) for b in boards_to_train])}")
        
        # Prepare training data
        tiles_data = []
        for board_idx in boards_to_train:
            board_labels = self.labels[board_idx]
            board = Board(int(board_idx))
            for tile_idx, label in board_labels.items():
                tile = board.get_tile_from_index(int(tile_idx))
                tiles_data.append({
                    "image": tile.image,
                    "tile_class": label["tile_class"],
                    "crown_count": label["crown_count"],
                    "board_idx": int(board_idx),
                    "tile_idx": int(tile_idx)
                })
        
        print(f"Training on {len(tiles_data)} tiles from {len(boards_to_train)} boards")
        
        # Create dataset and dataloader
        dataset = TileDataset(tiles_data, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = MultiTaskTileClassifier(self.num_classes, self.max_crowns)
        self.model.to(self.device)
        
        # Loss functions and optimizer
        criterion_class = nn.CrossEntropyLoss()
        criterion_crown = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=_SETTINGS["training"]["learning_rate"])
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            correct_class = 0
            correct_crown = 0
            total_samples = 0
            
            for batch in dataloader:
                images = batch["image"].to(self.device)
                tile_classes = batch["tile_class"].to(self.device)
                crown_counts = batch["crown_count"].to(self.device)
                
                optimizer.zero_grad()
                
                class_out, crown_out = self.model(images)
                
                loss_class = criterion_class(class_out, tile_classes)
                loss_crown = criterion_crown(crown_out, crown_counts)
                loss = loss_class + loss_crown
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, pred_class = torch.max(class_out, 1)
                _, pred_crown = torch.max(crown_out, 1)
                correct_class += (pred_class == tile_classes).sum().item()
                correct_crown += (pred_crown == crown_counts).sum().item()
                total_samples += images.size(0)
            
            avg_loss = total_loss / len(dataloader)
            class_acc = 100 * correct_class / total_samples
            crown_acc = 100 * correct_crown / total_samples
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - "
                  f"Class Acc: {class_acc:.2f}% - Crown Acc: {crown_acc:.2f}%")
        
        # Save model
        model_path = self.model_dir / "tile_classifier.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"\n✓ Model saved to {model_path}")
    
    def predict_board(self, board_idx):
        """Predict labels for a board using trained model"""
        if self.model is None:
            model_path = self.model_dir / "tile_classifier.pth"
            if not model_path.exists():
                raise ValueError("No trained model found. Please train first.")
            
            self.model = MultiTaskTileClassifier(self.num_classes, self.max_crowns)
            self.model.load_state_dict(torch.load(model_path))
            self.model.to(self.device)
        
        self.model.eval()
        
        board = Board(int(board_idx))
        predictions = {}
        
        with torch.no_grad():
            for tile in board.tiles:
                # Prepare image
                image = Image.fromarray(tile.image[:, :, ::-1])
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                # Predict
                class_out, crown_out = self.model(image_tensor)
                class_probs = torch.nn.functional.softmax(class_out, dim=1)
                crown_probs = torch.nn.functional.softmax(crown_out, dim=1)
                
                class_conf, class_pred = torch.max(class_probs, 1)
                crown_conf, crown_pred = torch.max(crown_probs, 1)
                
                predictions[tile.index] = {
                    "tile_class": class_pred.item(),
                    "tile_class_conf": class_conf.item(),
                    "crown_count": crown_pred.item(),
                    "crown_count_conf": crown_conf.item(),
                    "col": tile.col,
                    "row": tile.row
                }
        
        return board, predictions
    
    def visualize_predictions(self, board, predictions):
        """Visualize predictions on board with color coding"""
        vis_image = board.image.copy()
        tile_height = vis_image.shape[0] // self.board_height
        tile_width = vis_image.shape[1] // self.board_width
        
        for tile_idx, pred in predictions.items():
            col = pred["col"]
            row = pred["row"]
            
            # Calculate tile boundaries
            x1 = col * tile_width
            y1 = row * tile_height
            x2 = (col + 1) * tile_width
            y2 = (row + 1) * tile_height
            
            # Color based on confidence (green = high, yellow = medium, red = low)
            conf = min(pred["tile_class_conf"], pred["crown_count_conf"])
            if conf > 0.8:
                color = (0, 255, 0)  # Green
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)
            
            # Add text
            class_name = self.tile_classes[pred["tile_class"]]
            text = f"{class_name[:3]} C{pred["crown_count"]}"
            cv2.putText(vis_image, text, (x1+5, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis_image
    
    def review_and_correct_board(self, board_idx):
        """Show predictions and allow corrections using GUI"""
        print(f"\n=== Reviewing Board {board_idx} ===")
        
        board, predictions = self.predict_board(board_idx)
        
        # Use CustomTkinter GUI labeller for review
        gui = CTkGUILabeller(board, predictions, self.tile_classes, self.max_crowns)
        board_labels = gui.run()
        
        if board_labels is None:
            # User cancelled, fall back to manual labeling
            print("Falling back to manual labeling...")
            self.manual_label_board(board_idx)
            return False
        
        # Save the labels
        self.labels[str(board_idx)] = board_labels
        self._save_labels()
        print(f"✓ Board {board_idx} labels saved!")
        return True
    
    def run_auto_labelling(self, total_boards=74, training_boards_num=None):
        """Main workflow - automatically uses all boards already in labels file
        
        Args:
            total_boards: Total number of boards to label
            training_boards_num: Optional - randomly sample this many boards for training (for debugging)
                                If None, uses all labeled boards. Example: training_boards_num=5
        """
        print("=== Auto-Labelling Workflow ===")
        
        # Get already labeled boards
        existing_labeled = self.get_manually_labeled_boards()
        print(f"\nCurrently labeled boards: {existing_labeled}")
        
        # Verify we have enough training data
        if len(self.labels) < 2:
            print("\nError: Need at least 2 manually labeled boards to train.")
            print("Please manually label more boards first.")
            return
        
        print(f"\n✓ {len(self.labels)} boards available for training")
        
        # Store training_boards_num for use in retraining
        self.training_boards_num = training_boards_num
        
        # Train model
        self.train_model(num_epochs=_SETTINGS["training"]["initial_epochs"], training_boards_num=training_boards_num)
        
        # Auto-label remaining boards with review
        print("\n=== Starting Auto-Labelling with Human Review ===")
        new_labels_count = 0
        retrain_interval = _SETTINGS["training"]["retrain_interval"]
        
        for board_idx in range(1, total_boards + 1):
            if str(board_idx) not in self.labels:
                accepted = self.review_and_correct_board(board_idx)
                new_labels_count += 1
                
                # Check if we"re done BEFORE retraining
                boards_remaining = total_boards - int(board_idx)
                
                # Retrain every N new boards (but skip if we"re done)
                if new_labels_count % retrain_interval == 0 and boards_remaining > 0:
                    print(f"\n--- Retraining with new data (every {retrain_interval} boards) ---")
                    self.train_model(num_epochs=_SETTINGS["training"]["retrain_epochs"], training_boards_num=training_boards_num)
        
        print(f"\n✓✓✓ All {total_boards} boards labeled! ✓✓✓")
        print(f"Labels saved to: {self.labels_file}")

# Usage example
if __name__ == "__main__":
    labeller = AutoLabeller()
    
    # Option 1: Specify manually labeled boards
    labeller.run_auto_labelling(manually_labeled_boards=[1, 13, 14], total_boards=74)
    
    # Option 2: Let it prompt you interactively
    # labeller.run_auto_labelling(total_boards=74)