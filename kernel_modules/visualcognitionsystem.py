# sully_engine/kernel_modules/visual_cognition.py
# 🧠 Sully's Visual Cognition System - Understanding and reasoning about visual input

from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json
import uuid
import base64
from datetime import datetime
import io
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import numpy as np
import re
import logging
import tempfile
import requests
from pathlib import Path

# Import computer vision libraries
try:
    import cv2
    import torch
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.transforms import functional as F
    HAS_TORCH = True
except ImportError:
    logging.warning("PyTorch or torchvision not available. Using fallback object detection.")
    HAS_TORCH = False

try:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification, DetrImageProcessor, DetrForObjectDetection
    HAS_TRANSFORMERS = True
except ImportError:
    logging.warning("Transformers library not available. Using fallback for scene classification.")
    HAS_TRANSFORMERS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VisualObject:
    """
    Represents a detected object in a visual scene.
    """
    
    def __init__(self, label: str, confidence: float, 
               bbox: Optional[List[float]] = None,
               attributes: Optional[Dict[str, Any]] = None):
        """
        Initialize a visual object.
        
        Args:
            label: Object class label
            confidence: Detection confidence (0.0-1.0)
            bbox: Bounding box coordinates [x1, y1, x2, y2] normalized to 0-1
            attributes: Additional object attributes
        """
        self.label = label
        self.confidence = confidence
        self.bbox = bbox or [0.0, 0.0, 0.0, 0.0]
        self.attributes = attributes or {}
        self.relationships = []  # Relationships to other objects
        self.object_id = str(uuid.uuid4())
        
    def add_relationship(self, relation_type: str, target_object: 'VisualObject',
                       confidence: float = 1.0) -> None:
        """
        Add a relationship to another object.
        
        Args:
            relation_type: Type of relationship (e.g., "above", "contains", "next_to")
            target_object: The related object
            confidence: Relationship confidence (0.0-1.0)
        """
        self.relationships.append({
            "type": relation_type,
            "target_id": target_object.object_id,
            "target_label": target_object.label,
            "confidence": confidence
        })
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "object_id": self.object_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "attributes": self.attributes,
            "relationships": self.relationships
        }


class VisualScene:
    """
    Represents a complete visual scene with objects and their relationships.
    """
    
    def __init__(self, scene_id: Optional[str] = None):
        """
        Initialize a visual scene.
        
        Args:
            scene_id: Optional scene identifier
        """
        self.scene_id = scene_id or str(uuid.uuid4())
        self.objects = {}  # object_id -> VisualObject
        self.global_attributes = {}
        self.creation_time = datetime.now()
        self.source_image = None  # Could store path or reference to source
        self.width = 0
        self.height = 0
        
    def add_object(self, visual_object: VisualObject) -> None:
        """
        Add an object to the scene.
        
        Args:
            visual_object: Object to add
        """
        self.objects[visual_object.object_id] = visual_object
        
    def get_object_by_id(self, object_id: str) -> Optional[VisualObject]:
        """
        Get an object by ID.
        
        Args:
            object_id: Object identifier
            
        Returns:
            The object or None if not found
        """
        return self.objects.get(object_id)
        
    def get_objects_by_label(self, label: str) -> List[VisualObject]:
        """
        Get objects by label.
        
        Args:
            label: Object label to find
            
        Returns:
            List of matching objects
        """
        return [obj for obj in self.objects.values() if obj.label.lower() == label.lower()]
        
    def set_dimensions(self, width: int, height: int) -> None:
        """
        Set scene dimensions.
        
        Args:
            width: Scene width
            height: Scene height
        """
        self.width = width
        self.height = height
        
    def set_source(self, source: str) -> None:
        """
        Set source image reference.
        
        Args:
            source: Source image reference
        """
        self.source_image = source
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "scene_id": self.scene_id,
            "creation_time": self.creation_time.isoformat(),
            "width": self.width,
            "height": self.height,
            "source_image": self.source_image,
            "global_attributes": self.global_attributes,
            "objects": [obj.to_dict() for obj in self.objects.values()]
        }
        
    def describe(self) -> str:
        """
        Generate a textual description of the scene.
        
        Returns:
            Scene description
        """
        # Count objects by type
        object_counts = {}
        for obj in self.objects.values():
            label = obj.label
            if label not in object_counts:
                object_counts[label] = 0
            object_counts[label] += 1
            
        # Generate overall description
        description = f"Visual scene with {len(self.objects)} objects: "
        
        # List objects by type
        object_descriptions = []
        for label, count in object_counts.items():
            if count == 1:
                object_descriptions.append(f"1 {label}")
            else:
                object_descriptions.append(f"{count} {label}s")
                
        description += ", ".join(object_descriptions)
        
        # Add spatial relationships if available
        if any(obj.relationships for obj in self.objects.values()):
            description += ". Key relationships: "
            
            # Get top relationships
            relationships = []
            for obj in self.objects.values():
                for rel in obj.relationships:
                    if rel["confidence"] > 0.7:  # Only include high-confidence relationships
                        relationships.append(
                            f"{obj.label} {rel['type']} {rel['target_label']}"
                        )
                        
            # Add top relationships to description
            if relationships:
                description += ", ".join(relationships[:3])
                if len(relationships) > 3:
                    description += f" and {len(relationships) - 3} more"
                    
        return description
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'VisualScene':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Created visual scene
        """
        scene = VisualScene(scene_id=data.get("scene_id"))
        
        # Set basic properties
        scene.global_attributes = data.get("global_attributes", {})
        scene.width = data.get("width", 0)
        scene.height = data.get("height", 0)
        scene.source_image = data.get("source_image")
        
        # Try to parse creation time
        if "creation_time" in data:
            try:
                scene.creation_time = datetime.fromisoformat(data["creation_time"])
            except Exception:
                scene.creation_time = datetime.now()
                
        # Create objects
        for obj_data in data.get("objects", []):
            obj = VisualObject(
                label=obj_data.get("label", "unknown"),
                confidence=obj_data.get("confidence", 1.0),
                bbox=obj_data.get("bbox"),
                attributes=obj_data.get("attributes")
            )
            
            # Set object ID if available
            if "object_id" in obj_data:
                obj.object_id = obj_data["object_id"]
                
            # Set relationships
            obj.relationships = obj_data.get("relationships", [])
            
            # Add to scene
            scene.add_object(obj)
            
        return scene


class ObjectRecognitionModule:
    """
    Module for recognizing objects in images using modern computer vision techniques.
    """
    
    # COCO dataset class names for the Faster R-CNN model
    COCO_CLASSES = [
        'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
        'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the object recognition module.
        
        Args:
            model_path: Optional path to a custom recognition model
        """
        self.model_path = model_path
        self.initialized = False
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if HAS_TORCH else None
        self.color_analyzer = ColorAnalyzer()
        
        # Initialize model if PyTorch is available
        if HAS_TORCH:
            self.load_model()
            
    def load_model(self) -> bool:
        """
        Load the recognition model.
        
        Returns:
            Success indicator
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available. Object detection will use fallback implementation.")
            self.initialized = False
            return False
            
        try:
            if self.model_path:
                # Load custom model if path provided
                self.model = torch.load(self.model_path, map_location=self.device)
            else:
                # Use pre-trained Faster R-CNN model
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                self.model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.5)
                
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True
            logger.info(f"Object recognition model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load object recognition model: {str(e)}")
            self.initialized = False
            return False
    
    def _prepare_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> Optional[Image.Image]:
        """
        Prepare and normalize image for processing.
        
        Args:
            image_input: Input image (path, PIL Image, or numpy array)
            
        Returns:
            Prepared PIL image or None if preparation fails
        """
        try:
            if isinstance(image_input, str):
                # Handle URLs and local file paths
                if image_input.startswith(('http://', 'https://')):
                    response = requests.get(image_input, stream=True)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                elif os.path.exists(image_input):
                    image = Image.open(image_input).convert("RGB")
                else:
                    raise ValueError(f"Image path not found: {image_input}")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(np.uint8(image_input)).convert("RGB")
            else:
                raise ValueError("Unsupported image type")
                
            return image
        except Exception as e:
            logger.error(f"Image preparation failed: {str(e)}")
            return None
            
    def detect_objects(self, image: Union[str, Image.Image, np.ndarray]) -> List[VisualObject]:
        """
        Detect objects in an image using a deep learning model.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            List of detected objects
        """
        # Prepare image
        img = self._prepare_image(image)
        if img is None:
            return []
            
        # Get image dimensions
        width, height = img.size
        
        # Use PyTorch model if available
        if self.initialized and HAS_TORCH:
            try:
                # Convert image to tensor
                img_tensor = F.to_tensor(img).unsqueeze(0).to(self.device)
                
                # Perform inference
                with torch.no_grad():
                    predictions = self.model(img_tensor)
                    
                # Process results
                detected_objects = []
                
                for idx in range(len(predictions[0]['boxes'])):
                    score = predictions[0]['scores'][idx].item()
                    if score > 0.5:  # Confidence threshold
                        box = predictions[0]['boxes'][idx].cpu().numpy()
                        label_idx = predictions[0]['labels'][idx].item()
                        
                        # Get class label
                        if 0 <= label_idx < len(self.COCO_CLASSES):
                            label = self.COCO_CLASSES[label_idx]
                        else:
                            label = f"object_{label_idx}"
                        
                        # Normalize box coordinates
                        x1, y1, x2, y2 = box
                        norm_box = [
                            x1 / width,
                            y1 / height,
                            x2 / width,
                            y2 / height
                        ]
                        
                        # Extract object crop for attribute analysis
                        crop = img.crop((x1, y1, x2, y2))
                        
                        # Analyze attributes (like color)
                        attributes = {}
                        if label in ["car", "shirt", "book", "bicycle", "chair", "couch", "cup", "vase"]:
                            # These objects typically have distinctive colors
                            dominant_color, color_name = self.color_analyzer.analyze(crop)
                            attributes["color"] = color_name
                            attributes["color_rgb"] = dominant_color
                            
                        # Estimate size relative to image
                        area_ratio = ((x2 - x1) * (y2 - y1)) / (width * height)
                        if area_ratio < 0.05:
                            attributes["size"] = "small"
                        elif area_ratio < 0.15:
                            attributes["size"] = "medium"
                        else:
                            attributes["size"] = "large"
                            
                        # Create the visual object
                        obj = VisualObject(
                            label=label,
                            confidence=score,
                            bbox=norm_box,
                            attributes=attributes
                        )
                        
                        detected_objects.append(obj)
                        
                return detected_objects
                
            except Exception as e:
                logger.error(f"Object detection failed: {str(e)}")
                # Fall back to OpenCV-based detection if PyTorch fails
                return self._detect_with_opencv(img)
        else:
            # Use OpenCV-based detection as fallback
            return self._detect_with_opencv(img)
    
    def _detect_with_opencv(self, img: Image.Image) -> List[VisualObject]:
        """
        Fallback object detection using OpenCV.
        
        Args:
            img: PIL Image
            
        Returns:
            List of detected objects
        """
        detected_objects = []
        
        try:
            # Convert PIL Image to OpenCV format
            cv_img = np.array(img)
            cv_img = cv_img[:, :, ::-1].copy()  # RGB to BGR
            
            width, height = img.size
            
            # Use OpenCV's DNN module with a pre-trained model
            try:
                # Use YOLO or SSD if available
                config_file = os.environ.get("OPENCV_DNN_CONFIG", "")
                weights_file = os.environ.get("OPENCV_DNN_WEIGHTS", "")
                
                if os.path.exists(config_file) and os.path.exists(weights_file):
                    net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
                    layer_names = net.getLayerNames()
                    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
                    
                    # Prepare image for YOLO
                    blob = cv2.dnn.blobFromImage(cv_img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                    net.setInput(blob)
                    outs = net.forward(output_layers)
                    
                    class_ids = []
                    confidences = []
                    boxes = []
                    
                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.5:
                                # Object detected
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)
                                
                                # Rectangle coordinates
                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)
                                
                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)
                    
                    # Apply non-maximum suppression
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                    
                    for i in range(len(boxes)):
                        if i in indexes:
                            x, y, w, h = boxes[i]
                            label = self.COCO_CLASSES[class_ids[i]] if class_ids[i] < len(self.COCO_CLASSES) else f"object_{class_ids[i]}"
                            confidence = confidences[i]
                            
                            # Normalize coordinates
                            norm_box = [
                                max(0, x) / width,
                                max(0, y) / height,
                                min(width, x + w) / width,
                                min(height, y + h) / height
                            ]
                            
                            # Extract object crop for attribute analysis
                            crop = img.crop((max(0, x), max(0, y), min(width, x + w), min(height, y + h)))
                            
                            # Analyze attributes
                            attributes = {"size": "medium"}  # Default
                            
                            # Analyze color for certain objects
                            if label in ["car", "shirt", "book", "bicycle", "chair", "couch"]:
                                dominant_color, color_name = self.color_analyzer.analyze(crop)
                                attributes["color"] = color_name
                                
                            # Create visual object
                            obj = VisualObject(
                                label=label,
                                confidence=confidence,
                                bbox=norm_box,
                                attributes=attributes
                            )
                            
                            detected_objects.append(obj)
                else:
                    # Fall back to simple CV-based detection if YOLO not available
                    detected_objects = self._basic_cv_detection(cv_img, img)
            except Exception as e:
                logger.error(f"OpenCV DNN detection failed: {str(e)}")
                detected_objects = self._basic_cv_detection(cv_img, img)
                
            return detected_objects
                
        except Exception as e:
            logger.error(f"Fallback detection failed: {str(e)}")
            return []
    
    def _basic_cv_detection(self, cv_img, pil_img):
        """
        Basic detection using OpenCV traditional methods.
        """
        detected_objects = []
        width, height = pil_img.size
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Try to detect faces as a simple fallback
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Normalize coordinates
                norm_box = [
                    max(0, x) / width,
                    max(0, y) / height,
                    min(width, x + w) / width,
                    min(height, y + h) / height
                ]
                
                # Create visual object
                obj = VisualObject(
                    label="person",
                    confidence=0.8,
                    bbox=norm_box,
                    attributes={"size": "medium"}
                )
                
                detected_objects.append(obj)
                
            # Simple contour-based detection for other objects
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, threshold = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter small contours
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter out potential duplicates with faces
                    is_duplicate = False
                    for obj in detected_objects:
                        obj_x1, obj_y1, obj_x2, obj_y2 = [int(coord * (width if i % 2 == 0 else height)) 
                                                        for i, coord in enumerate(obj.bbox)]
                        overlap = max(0, min(x+w, obj_x2) - max(x, obj_x1)) * max(0, min(y+h, obj_y2) - max(y, obj_y1))
                        if overlap / (w * h) > 0.5:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        # Normalize coordinates
                        norm_box = [
                            max(0, x) / width,
                            max(0, y) / height,
                            min(width, x + w) / width,
                            min(height, y + h) / height
                        ]
                        
                        # Extract object crop for attribute analysis
                        crop = pil_img.crop((max(0, x), max(0, y), min(width, x + w), min(height, y + h)))
                        
                        # Analyze color
                        dominant_color, color_name = self.color_analyzer.analyze(crop)
                        
                        # Determine label based on shape analysis
                        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                        
                        if len(approx) == 4:
                            label = "book"  # Or could be box/table
                        elif len(approx) > 8:
                            label = "ball"
                        else:
                            # Default based on position
                            if y < height/3:
                                label = "lamp"
                            elif w > width/2:
                                label = "table"
                            else:
                                label = "object"
                                
                        # Create visual object
                        obj = VisualObject(
                            label=label,
                            confidence=0.6,
                            bbox=norm_box,
                            attributes={"color": color_name, "size": "medium"}
                        )
                        
                        detected_objects.append(obj)
                        
        except Exception as e:
            logger.error(f"Basic CV detection failed: {str(e)}")
            
        # If we still have no objects, add at least one generic object
        if not detected_objects:
            obj = VisualObject(
                label="object",
                confidence=0.5,
                bbox=[0.3, 0.3, 0.7, 0.7],
                attributes={"size": "medium"}
            )
            detected_objects.append(obj)
            
        return detected_objects
        
    def visualize_detections(self, image: Union[str, Image.Image], 
                           objects: List[VisualObject]) -> Image.Image:
        """
        Visualize detected objects on an image.
        
        Args:
            image: Input image (path or PIL Image)
            objects: Detected objects
            
        Returns:
            Annotated image
        """
        # Prepare image
        img = self._prepare_image(image)
        if img is None:
            # Return blank image on error
            return Image.new("RGB", (400, 300), color=(240, 240, 240))
            
        # Create drawing context
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Try to load a font
        try:
            font_path = os.path.join(os.path.dirname(__file__), "fonts", "arial.ttf")
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 14)
            else:
                # Try system fonts
                system_fonts = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                    "/Library/Fonts/Arial.ttf",  # macOS
                    "C:\\Windows\\Fonts\\arial.ttf"  # Windows
                ]
                for sys_font in system_fonts:
                    if os.path.exists(sys_font):
                        font = ImageFont.truetype(sys_font, 14)
                        break
                else:
                    font = ImageFont.load_default()
        except Exception as e:
            logger.warning(f"Failed to load font: {str(e)}")
            font = ImageFont.load_default()
            
        # Draw each detection
        for obj in objects:
            # Get normalized coordinates
            x1, y1, x2, y2 = obj.bbox
            
            # Convert to image coordinates
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            
            # Choose color based on confidence
            if obj.confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif obj.confidence > 0.5:
                color = (255, 255, 0)  # Yellow for medium confidence
            else:
                color = (255, 0, 0)  # Red for low confidence
                
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Prepare label text
            color_text = f" ({obj.attributes.get('color', '')})" if 'color' in obj.attributes else ""
            label_text = f"{obj.label}{color_text} ({obj.confidence:.2f})"
            
            # Measure text for background
            try:
                # Different versions of PIL have different APIs
                if hasattr(draw, "textbbox"):
                    text_bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                else:
                    text_width, text_height = draw.textsize(label_text, font=font)
            except Exception:
                # Fallback to estimated size
                text_width, text_height = len(label_text) * 7, 15
            
            # Draw background for text
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill=color
            )
            
            # Draw text
            draw.text((x1 + 2, y1 - text_height - 2), label_text, fill=(0, 0, 0), font=font)
            
        return img


class ColorAnalyzer:
    """
    Module for analyzing colors in images.
    """
    
    def __init__(self):
        """Initialize the color analyzer."""
        # Define color name mapping
        self.color_map = {
            # Red shades
            (255, 0, 0): "red",
            (180, 0, 0): "dark red",
            (255, 102, 102): "light red",
            (128, 0, 0): "maroon",
            # Green shades
            (0, 255, 0): "green",
            (0, 180, 0): "dark green",
            (102, 255, 102): "light green",
            (34, 139, 34): "forest green",
            # Blue shades
            (0, 0, 255): "blue",
            (0, 0, 180): "dark blue",
            (102, 102, 255): "light blue",
            (0, 0, 128): "navy",
            # Yellow shades
            (255, 255, 0): "yellow",
            (255, 215, 0): "gold",
            (255, 255, 102): "light yellow",
            # Purple shades
            (128, 0, 128): "purple",
            (186, 85, 211): "medium orchid",
            (153, 50, 204): "dark orchid",
            # Orange shades
            (255, 165, 0): "orange",
            (255, 127, 80): "coral",
            (255, 69, 0): "red-orange",
            # Brown shades
            (165, 42, 42): "brown",
            (210, 105, 30): "chocolate",
            (139, 69, 19): "saddle brown",
            # White and black
            (255, 255, 255): "white",
            (0, 0, 0): "black",
            # Gray shades
            (128, 128, 128): "gray",
            (169, 169, 169): "dark gray",
            (211, 211, 211): "light gray",
            # Pink shades
            (255, 192, 203): "pink",
            (255, 105, 180): "hot pink",
            (219, 112, 147): "pale violet red",
            # Cyan shades
            (0, 255, 255): "cyan",
            (224, 255, 255): "light cyan",
            (0, 206, 209): "dark turquoise",
        }
    
    def analyze(self, image: Image.Image) -> Tuple[Tuple[int, int, int], str]:
        """
        Analyze the dominant color in an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple containing (RGB tuple, color name)
        """
        try:
            # Resize image for faster processing
            img = image.copy()
            img.thumbnail((100, 100))
            
            # Convert to RGB if not already
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            # Get colors
            colors = img.getcolors(maxcolors=10000)
            if not colors:
                return (128, 128, 128), "gray"
                
            # Find most common color
            dominant_color = max(colors, key=lambda x: x[0])[1]
            
            # Find closest color name
            color_name = self._get_closest_color_name(dominant_color)
            
            return dominant_color, color_name
        except Exception as e:
            logger.error(f"Color analysis failed: {str(e)}")
            return (128, 128, 128), "gray"
    
    def _get_closest_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """
        Get the closest named color to an RGB value.
        
        Args:
            rgb: RGB color tuple
            
        Returns:
            Color name
        """
        min_distance = float('inf')
        closest_name = "unknown"
        
        for color_rgb, name in self.color_map.items():
            distance = self._color_distance(rgb, color_rgb)
            if distance < min_distance:
                min_distance = distance
                closest_name = name
                
        return closest_name
    
    def _color_distance(self, rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
        """
        Calculate distance between two RGB colors.
        
        Args:
            rgb1: First RGB color
            rgb2: Second RGB color
            
        Returns:
            Color distance
        """
        r1, g1, b1 = rgb1
        r2, g2, b2 = rgb2
        
        # Weighted Euclidean distance (human eye is more sensitive to green)
        return ((r2-r1)*0.30)**2 + ((g2-g1)*0.59)**2 + ((b2-b1)*0.11)**2


class SceneClassifier:
    """
    Module for classifying scene types using deep learning.
    """
    
    def __init__(self):
        """Initialize the scene classifier."""
        self.model = None
        self.processor = None
        self.initialized = False
        self.scene_categories = [
            "bathroom", "bedroom", "conference_room", "dining_room", "highway",
            "kitchen", "living_room", "mountain", "office", "street", "forest",
            "coast", "field", "store", "urban"
        ]
        
        # Initialize if transformers available
        if HAS_TRANSFORMERS:
            self.load_model()
    
    def load_model(self) -> bool:
        """
        Load scene classification model.
        
        Returns:
            Success indicator
        """
        if not HAS_TRANSFORMERS:
            logger.warning("Transformers library not available. Scene classification will use fallback.")
            return False
            
        try:
            # Load a pre-trained scene classification model
            # Most scene classification models use ResNet or ViT architecture
            self.processor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
            self.model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
            
            self.initialized = True
            logger.info("Scene classification model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load scene classification model: {str(e)}")
            self.initialized = False
            return False
    
    def classify(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify the scene type in an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with classification results
        """
        # Use transformers model if available
        if self.initialized and HAS_TRANSFORMERS:
            try:
                # Process image
                inputs = self.processor(images=image, return_tensors="pt")
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()
                
                # Map to category name
                categories = self.model.config.id2label
                category = categories[predicted_class_idx]
                confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()
                
                return {
                    "scene_type": category,
                    "confidence": confidence,
                    "method": "deep_learning"
                }
            except Exception as e:
                logger.error(f"Scene classification failed: {str(e)}")
                return self._fallback_classify(image)
        else:
            # Use fallback method
            return self._fallback_classify(image)
    
    def _fallback_classify(self, image: Image.Image) -> Dict[str, Any]:
        """
        Fallback scene classification using color and texture analysis.
        
        Args:
            image: PIL Image
            
        Returns:
            Classification results
        """
        try:
            # Resize for faster processing
            img = image.copy()
            img.thumbnail((200, 200))
            
            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            # Convert to numpy array
            img_array = np.array(img)
            
            # Simple color analysis
            avg_color = np.mean(img_array, axis=(0, 1))
            
            # Extract HSV for better color analysis
            hsv_img = None
            try:
                # Convert to HSV using OpenCV if available
                if 'cv2' in globals():
                    bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
            except Exception:
                hsv_img = None
                
            # Extract basic features
            brightness = np.mean(img_array)
            saturation = np.std(img_array)
            
            # Texture analysis (simple entropy)
            try:
                gray = np.mean(img_array, axis=2).astype(np.uint8)
                texture_entropy = np.std(gray)
            except Exception:
                texture_entropy = 0
                
            # Color histogram
            try:
                hist_r = np.histogram(img_array[:,:,0], bins=8, range=(0, 256))[0]
                hist_g = np.histogram(img_array[:,:,1], bins=8, range=(0, 256))[0]
                hist_b = np.histogram(img_array[:,:,2], bins=8, range=(0, 256))[0]
                
                # Normalize histograms
                hist_r = hist_r / np.sum(hist_r)
                hist_g = hist_g / np.sum(hist_g)
                hist_b = hist_b / np.sum(hist_b)
                
                # Combine histograms
                hist = np.concatenate([hist_r, hist_g, hist_b])
            except Exception:
                hist = np.ones(24) / 24
                
            # Simple rule-based classification
            # High blue often indicates outdoor scenes
            blue_ratio = avg_color[2] / (avg_color[0] + avg_color[1] + 1)
            
            # High green often indicates nature scenes
            green_ratio = avg_color[1] / (avg_color[0] + avg_color[2] + 1)
            
            # Determine scene type based on features
            if blue_ratio > 1.2 and brightness > 150:
                scene_type = "coast" if green_ratio < 0.8 else "mountain"
                confidence = 0.6
            elif green_ratio > 1.2:
                scene_type = "forest"
                confidence = 0.7
            elif brightness < 80:
                scene_type = "indoor" if texture_entropy > 40 else "urban"
                confidence = 0.5
            elif brightness > 200:
                scene_type = "snow" if np.std(img_array) < 40 else "beach"
                confidence = 0.6
            elif texture_entropy > 50:
                scene_type = "urban"
                confidence = 0.5
            else:
                scene_type = "indoor"
                confidence = 0.4
                
            return {
                "scene_type": scene_type,
                "confidence": confidence,
                "method": "color_analysis",
                "features": {
                    "brightness": float(brightness),
                    "texture": float(texture_entropy),
                    "blue_ratio": float(blue_ratio),
                    "green_ratio": float(green_ratio)
                }
            }
                
        except Exception as e:
            logger.error(f"Fallback scene classification failed: {str(e)}")
            return {
                "scene_type": "unknown",
                "confidence": 0.3,
                "method": "default"
            }


class SceneUnderstandingModule:
    """
    Module for understanding relationships and context in visual scenes.
    """
    
    def __init__(self):
        """Initialize the scene understanding module."""
        self.spatial_relationships = [
            "above", "below", "left_of", "right_of", "inside", "contains",
            "touching", "near", "far_from", "in_front_of", "behind",
            "centered", "aligned_with"
        ]
        self.scene_classifier = SceneClassifier()
        
    def analyze_scene(self, objects: List[VisualObject], 
                     image_width: int, image_height: int,
                     image: Optional[Image.Image] = None) -> VisualScene:
        """
        Analyze a scene with detected objects to understand relationships.
        
        Args:
            objects: Detected objects
            image_width: Image width
            image_height: Image height
            image: Optional original image for scene classification
            
        Returns:
            Analyzed visual scene
        """
        # Create a new scene
        scene = VisualScene()
        scene.set_dimensions(image_width, image_height)
        
        # Add objects to scene
        for obj in objects:
            scene.add_object(obj)
            
        # Analyze spatial relationships
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    # Identify relationships
                    relationships = self._identify_spatial_relationships(obj1, obj2)
                    
                    # Add relationships to object
                    for rel_type, confidence in relationships:
                        obj1.add_relationship(rel_type, obj2, confidence)
                        
        # Add global scene attributes
        scene.global_attributes["object_count"] = len(objects)
        scene.global_attributes["primary_objects"] = self._identify_primary_objects(objects)
        
        # Classify scene type
        if image is not None:
            # Use scene classifier if image is available
            classification = self.scene_classifier.classify(image)
            scene.global_attributes["scene_type"] = classification["scene_type"]
            scene.global_attributes["scene_confidence"] = classification["confidence"]
            scene.global_attributes["classification_method"] = classification["method"]
        else:
            # Fall back to object-based classification
            scene.global_attributes["scene_type"] = self._classify_scene_type(objects)
            scene.global_attributes["classification_method"] = "object_based"
            
        # Add additional scene metadata
        scene.global_attributes["analysis_time"] = datetime.now().isoformat()
        scene.global_attributes["object_density"] = len(objects) / (image_width * image_height / 1000000)  # Objects per million pixels
        
        # Detect potential activities
        scene.global_attributes["activities"] = self._detect_activities(objects)
        
        return scene
        
    def _identify_spatial_relationships(self, obj1: VisualObject, 
                                      obj2: VisualObject) -> List[Tuple[str, float]]:
        """
        Identify spatial relationships between two objects.
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            List of (relationship_type, confidence) tuples
        """
        relationships = []
        
        # Get bounding box coordinates
        x1_1, y1_1, x2_1, y2_1 = obj1.bbox
        x1_2, y1_2, x2_2, y2_2 = obj2.bbox
        
        # Calculate centers
        center_x1 = (x1_1 + x2_1) / 2
        center_y1 = (y1_1 + y2_1) / 2
        center_x2 = (x1_2 + x2_2) / 2
        center_y2 = (y1_2 + y2_2) / 2
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Check "above" relationship
        if y2_1 < y1_2:
            confidence = min(1.0, (y1_2 - y2_1) * 5)  # Higher confidence for larger vertical separation
            relationships.append(("above", confidence))
            
        # Check "below" relationship
        if y1_1 > y2_2:
            confidence = min(1.0, (y1_1 - y2_2) * 5)
            relationships.append(("below", confidence))
            
        # Check "left_of" relationship
        if x2_1 < x1_2:
            confidence = min(1.0, (x1_2 - x2_1) * 5)
            relationships.append(("left_of", confidence))
            
        # Check "right_of" relationship
        if x1_1 > x2_2:
            confidence = min(1.0, (x1_1 - x2_2) * 5)
            relationships.append(("right_of", confidence))
            
        # Check "contains" relationship
        if x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2:
            # Calculate containment ratio (area of obj2 / area of obj1)
            if area1 > 0:
                containment_ratio = area2 / area1
                # Higher confidence for smaller contained objects
                confidence = min(1.0, max(0.5, 1.0 - containment_ratio))
                relationships.append(("contains", confidence))
                
        # Check "inside" relationship
        if x1_2 <= x1_1 and y1_2 <= y1_1 and x2_2 >= x2_1 and y2_2 >= y2_1:
            # Calculate containment ratio (area of obj1 / area of obj2)
            if area2 > 0:
                containment_ratio = area1 / area2
                # Higher confidence for smaller contained objects
                confidence = min(1.0, max(0.5, 1.0 - containment_ratio))
                relationships.append(("inside", confidence))
                
        # Check "near" relationship
        distance = ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5
        if distance < 0.2:  # Threshold for "near"
            confidence = min(1.0, max(0.5, 1.0 - distance * 5))
            relationships.append(("near", confidence))
            
        # Check "touching" relationship
        # Simple approximation: consider objects touching if their bounding boxes are very close
        horizontal_overlap = (x1_1 <= x2_2 and x2_1 >= x1_2)
        vertical_overlap = (y1_1 <= y2_2 and y2_1 >= y1_2)
        
        if horizontal_overlap and vertical_overlap:
            # Calculate overlap area
            overlap_width = min(x2_1, x2_2) - max(x1_1, x1_2)
            overlap_height = min(y2_1, y2_2) - max(y1_1, y1_2)
            overlap_area = overlap_width * overlap_height
            
            # If overlap area is small, they might be touching
            if 0 < overlap_area < 0.05 * min(area1, area2):
                relationships.append(("touching", 0.7))
                
        # Check "aligned_with" relationship (horizontally or vertically)
        horizontal_aligned = abs(center_y1 - center_y2) < 0.05
        vertical_aligned = abs(center_x1 - center_x2) < 0.05
        
        if horizontal_aligned and not vertical_aligned:
            relationships.append(("horizontally_aligned_with", 0.8))
        elif vertical_aligned and not horizontal_aligned:
            relationships.append(("vertically_aligned_with", 0.8))
        elif horizontal_aligned and vertical_aligned:
            relationships.append(("centered_with", 0.9))
                
        return relationships
        
    def _identify_primary_objects(self, objects: List[VisualObject]) -> List[str]:
        """
        Identify primary objects in the scene.
        
        Args:
            objects: Detected objects
            
        Returns:
            List of primary object labels
        """
        if not objects:
            return []
            
        # Calculate importance score based on multiple factors
        object_importance = []
        
        for obj in objects:
            # Calculate area
            x1, y1, x2, y2 = obj.bbox
            area = (x2 - x1) * (y2 - y1)
            
            # Calculate centrality
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centrality = 1 - (((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5)
            
            # Consider confidence
            confidence = obj.confidence
            
            # Define intrinsic importance for certain object types
            intrinsic_importance = 1.0
            
            if obj.label == "person":
                intrinsic_importance = 1.5
            elif obj.label in ["car", "bicycle", "motorcycle", "truck", "bus", "train", "airplane"]:
                intrinsic_importance = 1.3
            elif obj.label in ["cat", "dog", "horse", "elephant", "bear", "giraffe", "zebra"]:
                intrinsic_importance = 1.2
                
            # Calculate overall importance
            importance = (area * 3 + centrality * 2 + confidence * 1.5) * intrinsic_importance
            
            object_importance.append((obj, importance))
            
        # Sort by importance score
        object_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 (or fewer if not enough objects)
        return [obj.label for obj, _ in object_importance[:min(3, len(object_importance))]]
        
    def _classify_scene_type(self, objects: List[VisualObject]) -> str:
        """
        Classify the type of scene based on objects.
        
        Args:
            objects: Detected objects
            
        Returns:
            Scene type classification
        """
        # Simple rule-based classification
        if not objects:
            return "unknown"
            
        # Count object types and group by categories
        object_types = {}
        category_counts = {
            "people": 0,
            "vehicles": 0,
            "furniture": 0,
            "nature": 0,
            "animals": 0,
            "food": 0,
            "electronics": 0,
            "kitchenware": 0,
            "sports": 0
        }
        
        # Define category mappings
        category_mappings = {
            "people": ["person"],
            "vehicles": ["car", "truck", "bus", "motorcycle", "bicycle", "train", "airplane", "boat"],
            "furniture": ["chair", "couch", "bed", "table", "desk", "bench", "bookshelf", "cabinet"],
            "nature": ["tree", "plant", "flower", "grass", "mountain", "rock", "water"],
            "animals": ["cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
            "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
            "electronics": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven"],
            "kitchenware": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "plate"],
            "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"]
        }
        
        # Count objects by type and category
        for obj in objects:
            # Count by specific type
            if obj.label not in object_types:
                object_types[obj.label] = 0
            object_types[obj.label] += 1
            
            # Count by category
            for category, labels in category_mappings.items():
                if obj.label in labels:
                    category_counts[category] += 1
                    break
                    
        # Determine the dominant category
        dominant_category = max(category_counts.items(), key=lambda x: x[1])
        
        # Apply classification rules
        if dominant_category[0] == "people" and dominant_category[1] >= 2:
            return "social"
            
        if dominant_category[0] == "vehicles" and dominant_category[1] >= 1:
            if "road" in object_types or "highway" in object_types:
                return "road"
            else:
                return "transportation"
                
        if dominant_category[0] == "furniture" and dominant_category[1] >= 1:
            if category_counts["kitchenware"] >= 1:
                return "kitchen"
            elif "bed" in object_types:
                return "bedroom"
            elif "couch" in object_types or "tv" in object_types:
                return "living_room"
            elif "dining table" in object_types:
                return "dining_room"
            else:
                return "indoor"
                
        if dominant_category[0] == "nature" and dominant_category[1] >= 1:
            if "tree" in object_types and object_types["tree"] >= 3:
                return "forest"
            elif "water" in object_types:
                return "coast"
            else:
                return "nature"
                
        if dominant_category[0] == "food" and dominant_category[1] >= 2:
            return "food"
            
        if dominant_category[0] == "animals" and dominant_category[1] >= 1:
            if category_counts["nature"] >= 1:
                return "wildlife"
            else:
                return "animal"
                
        if dominant_category[0] == "electronics" and category_counts["furniture"] >= 1:
            return "office"
            
        if dominant_category[0] == "sports" and dominant_category[1] >= 1:
            return "sports"
            
        # Default fallback based on object count
        if len(objects) == 0:
            return "empty"
        elif len(objects) <= 2:
            return "minimal"
        else:
            return "general"
    
    def _detect_activities(self, objects: List[VisualObject]) -> List[Dict[str, Any]]:
        """
        Detect potential activities in the scene based on object combinations.
        
        Args:
            objects: Detected objects
            
        Returns:
            List of potential activities
        """
        activities = []
        
        # Get object labels
        labels = [obj.label for obj in objects]
        label_count = {}
        for label in labels:
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
            
        # Define activity patterns
        activity_patterns = [
            {
                "name": "dining",
                "confidence": 0.8,
                "required": ["dining table"],
                "optional": ["chair", "fork", "knife", "spoon", "bowl", "cup", "wine glass", "person"]
            },
            {
                "name": "working",
                "confidence": 0.7,
                "required": ["laptop"],
                "optional": ["person", "chair", "desk", "book", "cup"]
            },
            {
                "name": "watching TV",
                "confidence": 0.7,
                "required": ["tv"],
                "optional": ["person", "couch", "remote"]
            },
            {
                "name": "cooking",
                "confidence": 0.7,
                "required": ["oven"],
                "optional": ["person", "microwave", "refrigerator", "sink", "bowl", "knife"]
            },
            {
                "name": "reading",
                "confidence": 0.6,
                "required": ["book"],
                "optional": ["person", "chair", "couch"]
            },
            {
                "name": "traveling",
                "confidence": 0.7,
                "required": ["car", "truck", "bus", "motorcycle", "train", "airplane"],  # Any one of these
                "optional": ["person", "suitcase", "backpack"]
            },
            {
                "name": "sports",
                "confidence": 0.7,
                "required": ["sports ball", "frisbee", "tennis racket", "baseball bat", "skateboard", "surfboard"],  # Any one of these
                "optional": ["person"]
            }
        ]
        
        # Check for each activity pattern
        for pattern in activity_patterns:
            # For required items, check if any of them is present (OR logic)
            required_present = False
            for required_item in pattern["required"]:
                if required_item in label_count:
                    required_present = True
                    break
                    
            if not required_present:
                continue
                
            # Count optional items
            optional_count = 0
            for optional_item in pattern["optional"]:
                if optional_item in label_count:
                    optional_count += 1
                    
            # Calculate confidence based on optional items
            optional_factor = min(1.0, optional_count / max(1, len(pattern["optional"])))
            confidence = pattern["confidence"] * (0.7 + 0.3 * optional_factor)
            
            # Add to detected activities
            activities.append({
                "activity": pattern["name"],
                "confidence": confidence
            })
            
        return activities
        
    def describe_scene(self, scene: VisualScene) -> str:
        """
        Generate a comprehensive description of a scene.
        
        Args:
            scene: The visual scene to describe
            
        Returns:
            Scene description
        """
        # Start with basic scene information
        scene_type = scene.global_attributes.get("scene_type", "general")
        object_count = scene.global_attributes.get("object_count", len(scene.objects))
        
        description = f"This appears to be a {scene_type} scene containing {object_count} objects. "
        
        # Mention primary objects
        primary_objects = scene.global_attributes.get("primary_objects", [])
        if primary_objects:
            description += f"The main elements are: {', '.join(primary_objects)}. "
            
        # Mention potential activities
        activities = scene.global_attributes.get("activities", [])
        if activities:
            # Filter by confidence and sort
            high_confidence_activities = [a for a in activities if a["confidence"] > 0.6]
            if high_confidence_activities:
                high_confidence_activities.sort(key=lambda x: x["confidence"], reverse=True)
                activity_names = [a["activity"] for a in high_confidence_activities[:2]]
                description += f"The scene suggests {' or '.join(activity_names)}. "
            
        # Describe spatial composition
        spatial_description = self._generate_spatial_description(scene)
        if spatial_description:
            description += spatial_description
            
        return description
        
    def _generate_spatial_description(self, scene: VisualScene) -> str:
        """
        Generate description of spatial relationships in the scene.
        
        Args:
            scene: The visual scene
            
        Returns:
            Spatial description
        """
        if not scene.objects:
            return ""
            
        # Find significant relationships
        significant_relations = []
        
        for obj_id, obj in scene.objects.items():
            for rel in obj.relationships:
                if rel["confidence"] > 0.7:  # Only high-confidence relationships
                    target_obj = scene.get_object_by_id(rel["target_id"])
                    if target_obj:
                        significant_relations.append(
                            (obj.label, rel["type"], target_obj.label, rel["confidence"])
                        )
                        
        # Sort by confidence
        significant_relations.sort(# sully_engine/kernel_modules/visual_cognition.py
# 🧠 Sully's Visual Cognition System - Understanding and reasoning about visual input

from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json
import uuid
import base64
from datetime import datetime
import io
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import numpy as np
import re
import logging
import tempfile
import requests
from pathlib import Path

# Import computer vision libraries
try:
    import cv2
    import torch
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.transforms import functional as F
    HAS_TORCH = True
except ImportError:
    logging.warning("PyTorch or torchvision not available. Using fallback object detection.")
    HAS_TORCH = False

try:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification, DetrImageProcessor, DetrForObjectDetection
    HAS_TRANSFORMERS = True
except ImportError:
    logging.warning("Transformers library not available. Using fallback for scene classification.")
    HAS_TRANSFORMERS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VisualObject:
    """
    Represents a detected object in a visual scene.
    """
    
    def __init__(self, label: str, confidence: float, 
               bbox: Optional[List[float]] = None,
               attributes: Optional[Dict[str, Any]] = None):
        """
        Initialize a visual object.
        
        Args:
            label: Object class label
            confidence: Detection confidence (0.0-1.0)
            bbox: Bounding box coordinates [x1, y1, x2, y2] normalized to 0-1
            attributes: Additional object attributes
        """
        self.label = label
        self.confidence = confidence
        self.bbox = bbox or [0.0, 0.0, 0.0, 0.0]
        self.attributes = attributes or {}
        self.relationships = []  # Relationships to other objects
        self.object_id = str(uuid.uuid4())
        
    def add_relationship(self, relation_type: str, target_object: 'VisualObject',
                       confidence: float = 1.0) -> None:
        """
        Add a relationship to another object.
        
        Args:
            relation_type: Type of relationship (e.g., "above", "contains", "next_to")
            target_object: The related object
            confidence: Relationship confidence (0.0-1.0)
        """
        self.relationships.append({
            "type": relation_type,
            "target_id": target_object.object_id,
            "target_label": target_object.label,
            "confidence": confidence
        })
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "object_id": self.object_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "attributes": self.attributes,
            "relationships": self.relationships
        }


class VisualScene:
    """
    Represents a complete visual scene with objects and their relationships.
    """
    
    def __init__(self, scene_id: Optional[str] = None):
        """
        Initialize a visual scene.
        
        Args:
            scene_id: Optional scene identifier
        """
        self.scene_id = scene_id or str(uuid.uuid4())
        self.objects = {}  # object_id -> VisualObject
        self.global_attributes = {}
        self.creation_time = datetime.now()
        self.source_image = None  # Could store path or reference to source
        self.width = 0
        self.height = 0
        
    def add_object(self, visual_object: VisualObject) -> None:
        """
        Add an object to the scene.
        
        Args:
            visual_object: Object to add
        """
        self.objects[visual_object.object_id] = visual_object
        
    def get_object_by_id(self, object_id: str) -> Optional[VisualObject]:
        """
        Get an object by ID.
        
        Args:
            object_id: Object identifier
            
        Returns:
            The object or None if not found
        """
        return self.objects.get(object_id)
        
    def get_objects_by_label(self, label: str) -> List[VisualObject]:
        """
        Get objects by label.
        
        Args:
            label: Object label to find
            
        Returns:
            List of matching objects
        """
        return [obj for obj in self.objects.values() if obj.label.lower() == label.lower()]
        
    def set_dimensions(self, width: int, height: int) -> None:
        """
        Set scene dimensions.
        
        Args:
            width: Scene width
            height: Scene height
        """
        self.width = width
        self.height = height
        
    def set_source(self, source: str) -> None:
        """
        Set source image reference.
        
        Args:
            source: Source image reference
        """
        self.source_image = source
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "scene_id": self.scene_id,
            "creation_time": self.creation_time.isoformat(),
            "width": self.width,
            "height": self.height,
            "source_image": self.source_image,
            "global_attributes": self.global_attributes,
            "objects": [obj.to_dict() for obj in self.objects.values()]
        }
        
    def describe(self) -> str:
        """
        Generate a textual description of the scene.
        
        Returns:
            Scene description
        """
        # Count objects by type
        object_counts = {}
        for obj in self.objects.values():
            label = obj.label
            if label not in object_counts:
                object_counts[label] = 0
            object_counts[label] += 1
            
        # Generate overall description
        description = f"Visual scene with {len(self.objects)} objects: "
        
        # List objects by type
        object_descriptions = []
        for label, count in object_counts.items():
            if count == 1:
                object_descriptions.append(f"1 {label}")
            else:
                object_descriptions.append(f"{count} {label}s")
                
        description += ", ".join(object_descriptions)
        
        # Add spatial relationships if available
        if any(obj.relationships for obj in self.objects.values()):
            description += ". Key relationships: "
            
            # Get top relationships
            relationships = []
            for obj in self.objects.values():
                for rel in obj.relationships:
                    if rel["confidence"] > 0.7:  # Only include high-confidence relationships
                        relationships.append(
                            f"{obj.label} {rel['type']} {rel['target_label']}"
                        )
                        
            # Add top relationships to description
            if relationships:
                description += ", ".join(relationships[:3])
                if len(relationships) > 3:
                    description += f" and {len(relationships) - 3} more"
                    
        return description
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'VisualScene':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Created visual scene
        """
        scene = VisualScene(scene_id=data.get("scene_id"))
        
        # Set basic properties
        scene.global_attributes = data.get("global_attributes", {})
        scene.width = data.get("width", 0)
        scene.height = data.get("height", 0)
        scene.source_image = data.get("source_image")
        
        # Try to parse creation time
        if "creation_time" in data:
            try:
                scene.creation_time = datetime.fromisoformat(data["creation_time"])
            except Exception:
                scene.creation_time = datetime.now()
                
        # Create objects
        for obj_data in data.get("objects", []):
            obj = VisualObject(
                label=obj_data.get("label", "unknown"),
                confidence=obj_data.get("confidence", 1.0),
                bbox=obj_data.get("bbox"),
                attributes=obj_data.get("attributes")
            )
            
            # Set object ID if available
            if "object_id" in obj_data:
                obj.object_id = obj_data["object_id"]
                
            # Set relationships
            obj.relationships = obj_data.get("relationships", [])
            
            # Add to scene
            scene.add_object(obj)
            
        return scene


class ObjectRecognitionModule:
    """
    Module for recognizing objects in images using modern computer vision techniques.
    """
    
    # COCO dataset class names for the Faster R-CNN model
    COCO_CLASSES = [
        'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
        'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the object recognition module.
        
        Args:
            model_path: Optional path to a custom recognition model
        """
        self.model_path = model_path
        self.initialized = False
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if HAS_TORCH else None
        self.color_analyzer = ColorAnalyzer()
        
        # Initialize model if PyTorch is available
        if HAS_TORCH:
            self.load_model()
            
    def load_model(self) -> bool:
        """
        Load the recognition model.
        
        Returns:
            Success indicator
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available. Object detection will use fallback implementation.")
            self.initialized = False
            return False
            
        try:
            if self.model_path:
                # Load custom model if path provided
                self.model = torch.load(self.model_path, map_location=self.device)
            else:
                # Use pre-trained Faster R-CNN model
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                self.model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.5)
                
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True
            logger.info(f"Object recognition model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load object recognition model: {str(e)}")
            self.initialized = False
            return False
    
    def _prepare_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> Optional[Image.Image]:
        """
        Prepare and normalize image for processing.
        
        Args:
            image_input: Input image (path, PIL Image, or numpy array)
            
        Returns:
            Prepared PIL image or None if preparation fails
        """
        try:
            if isinstance(image_input, str):
                # Handle URLs and local file paths
                if image_input.startswith(('http://', 'https://')):
                    response = requests.get(image_input, stream=True)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                elif os.path.exists(image_input):
                    image = Image.open(image_input).convert("RGB")
                else:
                    raise ValueError(f"Image path not found: {image_input}")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(np.uint8(image_input)).convert("RGB")
            else:
                raise ValueError("Unsupported image type")
                
            return image
        except Exception as e:
            logger.error(f"Image preparation failed: {str(e)}")
            return None
            
    def detect_objects(self, image: Union[str, Image.Image, np.ndarray]) -> List[VisualObject]:
        """
        Detect objects in an image using a deep learning model.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            List of detected objects
        """
        # Prepare image
        img = self._prepare_image(image)
        if img is None:
            return []
            
        # Get image dimensions
        width, height = img.size
        
        # Use PyTorch model if available
        if self.initialized and HAS_TORCH:
            try:
                # Convert image to tensor
                img_tensor = F.to_tensor(img).unsqueeze(0).to(self.device)
                
                # Perform inference
                with torch.no_grad():
                    predictions = self.model(img_tensor)
                    
                # Process results
                detected_objects = []
                
                for idx in range(len(predictions[0]['boxes'])):
                    score = predictions[0]['scores'][idx].item()
                    if score > 0.5:  # Confidence threshold
                        box = predictions[0]['boxes'][idx].cpu().numpy()
                        label_idx = predictions[0]['labels'][idx].item()
                        
                        # Get class label
                        if 0 <= label_idx < len(self.COCO_CLASSES):
                            label = self.COCO_CLASSES[label_idx]
                        else:
                            label = f"object_{label_idx}"
                        
                        # Normalize box coordinates
                        x1, y1, x2, y2 = box
                        norm_box = [
                            x1 / width,
                            y1 / height,
                            x2 / width,
                            y2 / height
                        ]
                        
                        # Extract object crop for attribute analysis
                        crop = img.crop((x1, y1, x2, y2))
                        
                        # Analyze attributes (like color)
                        attributes = {}
                        if label in ["car", "shirt", "book", "bicycle", "chair", "couch", "cup", "vase"]:
                            # These objects typically have distinctive colors
                            dominant_color, color_name = self.color_analyzer.analyze(crop)
                            attributes["color"] = color_name
                            attributes["color_rgb"] = dominant_color
                            
                        # Estimate size relative to image
                        area_ratio = ((x2 - x1) * (y2 - y1)) / (width * height)
                        if area_ratio < 0.05:
                            attributes["size"] = "small"
                        elif area_ratio < 0.15:
                            attributes["size"] = "medium"
                        else:
                            attributes["size"] = "large"
                            
                        # Create the visual object
                        obj = VisualObject(
                            label=label,
                            confidence=score,
                            bbox=norm_box,
                            attributes=attributes
                        )
                        
                        detected_objects.append(obj)
                        
                return detected_objects
                
            except Exception as e:
                logger.error(f"Object detection failed: {str(e)}")
                # Fall back to OpenCV-based detection if PyTorch fails
                return self._detect_with_opencv(img)
        else:
            # Use OpenCV-based detection as fallback
            return self._detect_with_opencv(img)
    
    def _detect_with_opencv(self, img: Image.Image) -> List[VisualObject]:
        """
        Fallback object detection using OpenCV.
        
        Args:
            img: PIL Image
            
        Returns:
            List of detected objects
        """
        detected_objects = []
        
        try:
            # Convert PIL Image to OpenCV format
            cv_img = np.array(img)
            cv_img = cv_img[:, :, ::-1].copy()  # RGB to BGR
            
            width, height = img.size
            
            # Use OpenCV's DNN module with a pre-trained model
            try:
                # Use YOLO or SSD if available
                config_file = os.environ.get("OPENCV_DNN_CONFIG", "")
                weights_file = os.environ.get("OPENCV_DNN_WEIGHTS", "")
                
                if os.path.exists(config_file) and os.path.exists(weights_file):
                    net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
                    layer_names = net.getLayerNames()
                    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
                    
                    # Prepare image for YOLO
                    blob = cv2.dnn.blobFromImage(cv_img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                    net.setInput(blob)
                    outs = net.forward(output_layers)
                    
                    class_ids = []
                    confidences = []
                    boxes = []
                    
                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.5:
                                # Object detected
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)
                                
                                # Rectangle coordinates
                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)
                                
                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)
                    
                    # Apply non-maximum suppression
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                    
                    for i in range(len(boxes)):
                        if i in indexes:
                            x, y, w, h = boxes[i]
                            label = self.COCO_CLASSES[class_ids[i]] if class_ids[i] < len(self.COCO_CLASSES) else f"object_{class_ids[i]}"
                            confidence = confidences[i]
                            
                            # Normalize coordinates
                            norm_box = [
                                max(0, x) / width,
                                max(0, y) / height,
                                min(width, x + w) / width,
                                min(height, y + h) / height
                            ]
                            
                            # Extract object crop for attribute analysis
                            crop = img.crop((max(0, x), max(0, y), min(width, x + w), min(height, y + h)))
                            
                            # Analyze attributes
                            attributes = {"size": "medium"}  # Default
                            
                            # Analyze color for certain objects
                            if label in ["car", "shirt", "book", "bicycle", "chair", "couch"]:
                                dominant_color, color_name = self.color_analyzer.analyze(crop)
                                attributes["color"] = color_name
                                
                            # Create visual object
                            obj = VisualObject(
                                label=label,
                                confidence=confidence,
                                bbox=norm_box,
                                attributes=attributes
                            )
                            
                            detected_objects.append(obj)
                else:
                    # Fall back to simple CV-based detection if YOLO not available
                    detected_objects = self._basic_cv_detection(cv_img, img)
            except Exception as e:
                logger.error(f"OpenCV DNN detection failed: {str(e)}")
                detected_objects = self._basic_cv_detection(cv_img, img)
                
            return detected_objects
                
        except Exception as e:
            logger.error(f"Fallback detection failed: {str(e)}")
            return []
    
    def _basic_cv_detection(self, cv_img, pil_img):
        """
        Basic detection using OpenCV traditional methods.
        """
        detected_objects = []
        width, height = pil_img.size
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Try to detect faces as a simple fallback
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Normalize coordinates
                norm_box = [
                    max(0, x) / width,
                    max(0, y) / height,
                    min(width, x + w) / width,
                    min(height, y + h) / height
                ]
                
                # Create visual object
                obj = VisualObject(
                    label="person",
                    confidence=0.8,
                    bbox=norm_box,
                    attributes={"size": "medium"}
                )
                
                detected_objects.append(obj)
                
            # Simple contour-based detection for other objects
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, threshold = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter small contours
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter out potential duplicates with faces
                    is_duplicate = False
                    for obj in detected_objects:
                        obj_x1, obj_y1, obj_x2, obj_y2 = [int(coord * (width if i % 2 == 0 else height)) 
                                                        for i, coord in enumerate(obj.bbox)]
                        overlap = max(0, min(x+w, obj_x2) - max(x, obj_x1)) * max(0, min(y+h, obj_y2) - max(y, obj_y1))
                        if overlap / (w * h) > 0.5:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        # Normalize coordinates
                        norm_box = [
                            max(0, x) / width,
                            max(0, y) / height,
                            min(width, x + w) / width,
                            min(height, y + h) / height
                        ]
                        
                        # Extract object crop for attribute analysis
                        crop = pil_img.crop((max(0, x), max(0, y), min(width, x + w), min(height, y + h)))
                        
                        # Analyze color
                        dominant_color, color_name = self.color_analyzer.analyze(crop)
                        
                        # Determine label based on shape analysis
                        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                        
                        if len(approx) == 4:
                            label = "book"  # Or could be box/table
                        elif len(approx) > 8:
                            label = "ball"
                        else:
                            # Default based on position
                            if y < height/3:
                                label = "lamp"
                            elif w > width/2:
                                label = "table"
                            else:
                                label = "object"
                                
                        # Create visual object
                        obj = VisualObject(
                            label=label,
                            confidence=0.6,
                            bbox=norm_box,
                            attributes={"color": color_name, "size": "medium"}
                        )
                        
                        detected_objects.append(obj)
                        
        except Exception as e:
            logger.error(f"Basic CV detection failed: {str(e)}")
            
        # If we still have no objects, add at least one generic object
        if not detected_objects:
            obj = VisualObject(
                label="object",
                confidence=0.5,
                bbox=[0.3, 0.3, 0.7, 0.7],
                attributes={"size": "medium"}
            )
            detected_objects.append(obj)
            
        return detected_objects
        
    def visualize_detections(self, image: Union[str, Image.Image], 
                           objects: List[VisualObject]) -> Image.Image:
        """
        Visualize detected objects on an image.
        
        Args:
            image: Input image (path or PIL Image)
            objects: Detected objects
            
        Returns:
            Annotated image
        """
        # Prepare image
        img = self._prepare_image(image)
        if img is None:
            # Return blank image on error
            return Image.new("RGB", (400, 300), color=(240, 240, 240))
            
        # Create drawing context
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Try to load a font
        try:
            font_path = os.path.join(os.path.dirname(__file__), "fonts", "arial.ttf")
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 14)
            else:
                # Try system fonts
                system_fonts = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                    "/Library/Fonts/Arial.ttf",  # macOS
                    "C:\\Windows\\Fonts\\arial.ttf"  # Windows
                ]
                for sys_font in system_fonts:
                    if os.path.exists(sys_font):
                        font = ImageFont.truetype(sys_font, 14)
                        break
                else:
                    font = ImageFont.load_default()
        except Exception as e:
            logger.warning(f"Failed to load font: {str(e)}")
            font = ImageFont.load_default()
            
        # Draw each detection
        for obj in objects:
            # Get normalized coordinates
            x1, y1, x2, y2 = obj.bbox
            
            # Convert to image coordinates
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            
            # Choose color based on confidence
            if obj.confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif obj.confidence > 0.5:
                color = (255, 255, 0)  # Yellow for medium confidence
            else:
                color = (255, 0, 0)  # Red for low confidence
                
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Prepare label text
            color_text = f" ({obj.attributes.get('color', '')})" if 'color' in obj.attributes else ""
            label_text = f"{obj.label}{color_text} ({obj.confidence:.2f})"
            
            # Measure text for background
            try:
                # Different versions of PIL have different APIs
                if hasattr(draw, "textbbox"):
                    text_bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                else:
                    text_width, text_height = draw.textsize(label_text, font=font)
            except Exception:
                # Fallback to estimated size
                text_width, text_height = len(label_text) * 7, 15
            
            # Draw background for text
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill=color
            )
            
            # Draw text
            draw.text((x1 + 2, y1 - text_height - 2), label_text, fill=(0, 0, 0), font=font)
            
        return img


class ColorAnalyzer:
    """
    Module for analyzing colors in images.
    """
    
    def __init__(self):
        """Initialize the color analyzer."""
        # Define color name mapping
        self.color_map = {
            # Red shades
            (255, 0, 0): "red",
            (180, 0, 0): "dark red",
            (255, 102, 102): "light red",
            (128, 0, 0): "maroon",
            # Green shades
            (0, 255, 0): "green",
            (0, 180, 0): "dark green",
            (102, 255, 102): "light green",
            (34, 139, 34): "forest green",
            # Blue shades
            (0, 0, 255): "blue",
            (0, 0, 180): "dark blue",
            (102, 102, 255): "light blue",
            (0, 0, 128): "navy",
            # Yellow shades
            (255, 255, 0): "yellow",
            (255, 215, 0): "gold",
            (255, 255, 102): "light yellow",
            # Purple shades
            (128, 0, 128): "purple",
            (186, 85, 211): "medium orchid",
            (153, 50, 204): "dark orchid",
            # Orange shades
            (255, 165, 0): "orange",
            (255, 127, 80): "coral",
            (255, 69, 0): "red-orange",
            # Brown shades
            (165, 42, 42): "brown",
            (210, 105, 30): "chocolate",
            (139, 69, 19): "saddle brown",
            # White and black
            (255, 255, 255): "white",
            (0, 0, 0): "black",
            # Gray shades
            (128, 128, 128): "gray",
            (169, 169, 169): "dark gray",
            (211, 211, 211): "light gray",
            # Pink shades
            (255, 192, 203): "pink",
            (255, 105, 180): "hot pink",
            (219, 112, 147): "pale violet red",
            # Cyan shades
            (0, 255, 255): "cyan",
            (224, 255, 255): "light cyan",
            (0, 206, 209): "dark turquoise",
        }
    
    def analyze(self, image: Image.Image) -> Tuple[Tuple[int, int, int], str]:
        """
        Analyze the dominant color in an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple containing (RGB tuple, color name)
        """
        try:
            # Resize image for faster processing
            img = image.copy()
            img.thumbnail((100, 100))
            
            # Convert to RGB if not already
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            # Get colors
            colors = img.getcolors(maxcolors=10000)
            if not colors:
                return (128, 128, 128), "gray"
                
            # Find most common color
            dominant_color = max(colors, key=lambda x: x[0])[1]
            
            # Find closest color name
            color_name = self._get_closest_color_name(dominant_color)
            
            return dominant_color, color_name
        except Exception as e:
            logger.error(f"Color analysis failed: {str(e)}")
            return (128, 128, 128), "gray"
    
    def _get_closest_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """
        Get the closest named color to an RGB value.
        
        Args:
            rgb: RGB color tuple
            
        Returns:
            Color name
        """
        min_distance = float('inf')
        closest_name = "unknown"
        
        for color_rgb, name in self.color_map.items():
            distance = self._color_distance(rgb, color_rgb)
            if distance < min_distance:
                min_distance = distance
                closest_name = name
                
        return closest_name
    
    def _color_distance(self, rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
        """
        Calculate distance between two RGB colors.
        
        Args:
            rgb1: First RGB color
            rgb2: Second RGB color
            
        Returns:
            Color distance
        """
        r1, g1, b1 = rgb1
        r2, g2, b2 = rgb2
        
        # Weighted Euclidean distance (human eye is more sensitive to green)
        return ((r2-r1)*0.30)**2 + ((g2-g1)*0.59)**2 + ((b2-b1)*0.11)**2


class SceneClassifier:
    """
    Module for classifying scene types using deep learning.
    """
    
    def __init__(self):
        """Initialize the scene classifier."""
        self.model = None
        self.processor = None
        self.initialized = False
        self.scene_categories = [
            "bathroom", "bedroom", "conference_room", "dining_room", "highway",
            "kitchen", "living_room", "mountain", "office", "street", "forest",
            "coast", "field", "store", "urban"
        ]
        
        # Initialize if transformers available
        if HAS_TRANSFORMERS:
            self.load_model()
    
    def load_model(self) -> bool:
        """
        Load scene classification model.
        
        Returns:
            Success indicator
        """
        if not HAS_TRANSFORMERS:
            logger.warning("Transformers library not available. Scene classification will use fallback.")
            return False
            
        try:
            # Load a pre-trained scene classification model
            # Most scene classification models use ResNet or ViT architecture
            self.processor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
            self.model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
            
            self.initialized = True
            logger.info("Scene classification model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load scene classification model: {str(e)}")
            self.initialized = False
            return False
    
    def classify(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify the scene type in an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with classification results
        """
        # Use transformers model if available
        if self.initialized and HAS_TRANSFORMERS:
            try:
                # Process image
                inputs = self.processor(images=image, return_tensors="pt")
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()
                
                # Map to category name
                categories = self.model.config.id2label
                category = categories[predicted_class_idx]
                confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()
                
                return {
                    "scene_type": category,
                    "confidence": confidence,
                    "method": "deep_learning"
                }
            except Exception as e:
                logger.error(f"Scene classification failed: {str(e)}")
                return self._fallback_classify(image)
        else:
            # Use fallback method
            return self._fallback_classify(image)
    
    def _fallback_classify(self, image: Image.Image) -> Dict[str, Any]:
        """
        Fallback scene classification using color and texture analysis.
        
        Args:
            image: PIL Image
            
        Returns:
            Classification results
        """
        try:
            # Resize for faster processing
            img = image.copy()
            img.thumbnail((200, 200))
            
            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            # Convert to numpy array
            img_array = np.array(img)
            
            # Simple color analysis
            avg_color = np.mean(img_array, axis=(0, 1))
            
            # Extract HSV for better color analysis
            hsv_img = None
            try:
                # Convert to HSV using OpenCV if available
                if 'cv2' in globals():
                    bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
            except Exception:
                hsv_img = None
                
            # Extract basic features
            brightness = np.mean(img_array)
            saturation = np.std(img_array)
            
            # Texture analysis (simple entropy)
            try:
                gray = np.mean(img_array, axis=2).astype(np.uint8)
                texture_entropy = np.std(gray)
            except Exception:
                texture_entropy = 0
                
            # Color histogram
            try:
                hist_r = np.histogram(img_array[:,:,0], bins=8, range=(0, 256))[0]
                hist_g = np.histogram(img_array[:,:,1], bins=8, range=(0, 256))[0]
                hist_b = np.histogram(img_array[:,:,2], bins=8, range=(0, 256))[0]
                
                # Normalize histograms
                hist_r = hist_r / np.sum(hist_r)
                hist_g = hist_g / np.sum(hist_g)
                hist_b = hist_b / np.sum(hist_b)
                
                # Combine histograms
                hist = np.concatenate([hist_r, hist_g, hist_b])
            except Exception:
                hist = np.ones(24) / 24
                
            # Simple rule-based classification
            # High blue often indicates outdoor scenes
            blue_ratio = avg_color[2] / (avg_color[0] + avg_color[1] + 1)
            
            # High green often indicates nature scenes
            green_ratio = avg_color[1] / (avg_color[0] + avg_color[2] + 1)
            
            # Determine scene type based on features
            if blue_ratio > 1.2 and brightness > 150:
                scene_type = "coast" if green_ratio < 0.8 else "mountain"
                confidence = 0.6
            elif green_ratio > 1.2:
                scene_type = "forest"
                confidence = 0.7
            elif brightness < 80:
                scene_type = "indoor" if texture_entropy > 40 else "urban"
                confidence = 0.5
            elif brightness > 200:
                scene_type = "snow" if np.std(img_array) < 40 else "beach"
                confidence = 0.6
            elif texture_entropy > 50:
                scene_type = "urban"
                confidence = 0.5
            else:
                scene_type = "indoor"
                confidence = 0.4
                
            return {
                "scene_type": scene_type,
                "confidence": confidence,
                "method": "color_analysis",
                "features": {
                    "brightness": float(brightness),
                    "texture": float(texture_entropy),
                    "blue_ratio": float(blue_ratio),
                    "green_ratio": float(green_ratio)
                }
            }
                
        except Exception as e:
            logger.error(f"Fallback scene classification failed: {str(e)}")
            return {
                "scene_type": "unknown",
                "confidence": 0.3,
                "method": "default"
            }


class SceneUnderstandingModule:
    """
    Module for understanding relationships and context in visual scenes.
    """
    
    def __init__(self):
        """Initialize the scene understanding module."""
        self.spatial_relationships = [
            "above", "below", "left_of", "right_of", "inside", "contains",
            "touching", "near", "far_from", "in_front_of", "behind",
            "centered", "aligned_with"
        ]
        self.scene_classifier = SceneClassifier()
        
    def analyze_scene(self, objects: List[VisualObject], 
                     image_width: int, image_height: int,
                     image: Optional[Image.Image] = None) -> VisualScene:
        """
        Analyze a scene with detected objects to understand relationships.
        
        Args:
            objects: Detected objects
            image_width: Image width
            image_height: Image height
            image: Optional original image for scene classification
            
        Returns:
            Analyzed visual scene
        """
        # Create a new scene
        scene = VisualScene()
        scene.set_dimensions(image_width, image_height)
        
        # Add objects to scene
        for obj in objects:
            scene.add_object(obj)
            
        # Analyze spatial relationships
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    # Identify relationships
                    relationships = self._identify_spatial_relationships(obj1, obj2)
                    
                    # Add relationships to object
                    for rel_type, confidence in relationships:
                        obj1.add_relationship(rel_type, obj2, confidence)
                        
        # Add global scene attributes
        scene.global_attributes["object_count"] = len(objects)
        scene.global_attributes["primary_objects"] = self._identify_primary_objects(objects)
        
        # Classify scene type
        if image is not None:
            # Use scene classifier if image is available
            classification = self.scene_classifier.classify(image)
            scene.global_attributes["scene_type"] = classification["scene_type"]
            scene.global_attributes["scene_confidence"] = classification["confidence"]
            scene.global_attributes["classification_method"] = classification["method"]
        else:
            # Fall back to object-based classification
            scene.global_attributes["scene_type"] = self._classify_scene_type(objects)
            scene.global_attributes["classification_method"] = "object_based"
            
        # Add additional scene metadata
        scene.global_attributes["analysis_time"] = datetime.now().isoformat()
        scene.global_attributes["object_density"] = len(objects) / (image_width * image_height / 1000000)  # Objects per million pixels
        
        # Detect potential activities
        scene.global_attributes["activities"] = self._detect_activities(objects)
        
        return scene
        
    def _identify_spatial_relationships(self, obj1: VisualObject, 
                                      obj2: VisualObject) -> List[Tuple[str, float]]:
        """
        Identify spatial relationships between two objects.
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            List of (relationship_type, confidence) tuples
        """
        relationships = []
        
        # Get bounding box coordinates
        x1_1, y1_1, x2_1, y2_1 = obj1.bbox
        x1_2, y1_2, x2_2, y2_2 = obj2.bbox
        
        # Calculate centers
        center_x1 = (x1_1 + x2_1) / 2
        center_y1 = (y1_1 + y2_1) / 2
        center_x2 = (x1_2 + x2_2) / 2
        center_y2 = (y1_2 + y2_2) / 2
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Check "above" relationship
        if y2_1 < y1_2:
            confidence = min(1.0, (y1_2 - y2_1) * 5)  # Higher confidence for larger vertical separation
            relationships.append(("above", confidence))
            
        # Check "below" relationship
        if y1_1 > y2_2:
            confidence = min(1.0, (y1_1 - y2_2) * 5)
            relationships.append(("below", confidence))
            
        # Check "left_of" relationship
        if x2_1 < x1_2:
            confidence = min(1.0, (x1_2 - x2_1) * 5)
            relationships.append(("left_of", confidence))
            
        # Check "right_of" relationship
        if x1_1 > x2_2:
            confidence = min(1.0, (x1_1 - x2_2) * 5)
            relationships.append(("right_of", confidence))
            
        # Check "contains" relationship
        if x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2:
            # Calculate containment ratio (area of obj2 / area of obj1)
            if area1 > 0:
                containment_ratio = area2 / area1
                # Higher confidence for smaller contained objects
                confidence = min(1.0, max(0.5, 1.0 - containment_ratio))
                relationships.append(("contains", confidence))
                
        # Check "inside" relationship
        if x1_2 <= x1_1 and y1_2 <= y1_1 and x2_2 >= x2_1 and y2_2 >= y2_1:
            # Calculate containment ratio (area of obj1 / area of obj2)
            if area2 > 0:
                containment_ratio = area1 / area2
                # Higher confidence for smaller contained objects
                confidence = min(1.0, max(0.5, 1.0 - containment_ratio))
                relationships.append(("inside", confidence))
                
        # Check "near" relationship
        distance = ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5
        if distance < 0.2:  # Threshold for "near"
            confidence = min(1.0, max(0.5, 1.0 - distance * 5))
            relationships.append(("near", confidence))
            
        # Check "touching" relationship
        # Simple approximation: consider objects touching if their bounding boxes are very close
        horizontal_overlap = (x1_1 <= x2_2 and x2_1 >= x1_2)
        vertical_overlap = (y1_1 <= y2_2 and y2_1 >= y1_2)
        
        if horizontal_overlap and vertical_overlap:
            # Calculate overlap area
            overlap_width = min(x2_1, x2_2) - max(x1_1, x1_2)
            overlap_height = min(y2_1, y2_2) - max(y1_1, y1_2)
            overlap_area = overlap_width * overlap_height
            
            # If overlap area is small, they might be touching
            if 0 < overlap_area < 0.05 * min(area1, area2):
                relationships.append(("touching", 0.7))
                
        # Check "aligned_with" relationship (horizontally or vertically)
        horizontal_aligned = abs(center_y1 - center_y2) < 0.05
        vertical_aligned = abs(center_x1 - center_x2) < 0.05
        
        if horizontal_aligned and not vertical_aligned:
            relationships.append(("horizontally_aligned_with", 0.8))
        elif vertical_aligned and not horizontal_aligned:
            relationships.append(("vertically_aligned_with", 0.8))
        elif horizontal_aligned and vertical_aligned:
            relationships.append(("centered_with", 0.9))
                
        return relationships
        
    def _identify_primary_objects(self, objects: List[VisualObject]) -> List[str]:
        """
        Identify primary objects in the scene.
        
        Args:
            objects: Detected objects
            
        Returns:
            List of primary object labels
        """
        if not objects:
            return []
            
        # Calculate importance score based on multiple factors
        object_importance = []
        
        for obj in objects:
            # Calculate area
            x1, y1, x2, y2 = obj.bbox
            area = (x2 - x1) * (y2 - y1)
            
            # Calculate centrality
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centrality = 1 - (((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5)
            
            # Consider confidence
            confidence = obj.confidence
            
            # Define intrinsic importance for certain object types
            intrinsic_importance = 1.0
            
            if obj.label == "person":
                intrinsic_importance = 1.5
            elif obj.label in ["car", "bicycle", "motorcycle", "truck", "bus", "train", "airplane"]:
                intrinsic_importance = 1.3
            elif obj.label in ["cat", "dog", "horse", "elephant", "bear", "giraffe", "zebra"]:
                intrinsic_importance = 1.2
                
            # Calculate overall importance
            importance = (area * 3 + centrality * 2 + confidence * 1.5) * intrinsic_importance
            
            object_importance.append((obj, importance))
            
        # Sort by importance score
        object_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 (or fewer if not enough objects)
        return [obj.label for obj, _ in object_importance[:min(3, len(object_importance))]]
        
    def _classify_scene_type(self, objects: List[VisualObject]) -> str:
        """
        Classify the type of scene based on objects.
        
        Args:
            objects: Detected objects
            
        Returns:
            Scene type classification
        """
        # Simple rule-based classification
        if not objects:
            return "unknown"
            
        # Count object types and group by categories
        object_types = {}
        category_counts = {
            "people": 0,
            "vehicles": 0,
            "furniture": 0,
            "nature": 0,
            "animals": 0,
            "food": 0,
            "electronics": 0,
            "kitchenware": 0,
            "sports": 0
        }
        
        # Define category mappings
        category_mappings = {
            "people": ["person"],
            "vehicles": ["car", "truck", "bus", "motorcycle", "bicycle", "train", "airplane", "boat"],
            "furniture": ["chair", "couch", "bed", "table", "desk", "bench", "bookshelf", "cabinet"],
            "nature": ["tree", "plant", "flower", "grass", "mountain", "rock", "water"],
            "animals": ["cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
            "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
            "electronics": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven"],
            "kitchenware": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "plate"],
            "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"]
        }
        
        # Count objects by type and category
        for obj in objects:
            # Count by specific type
            if obj.label not in object_types:
                object_types[obj.label] = 0
            object_types[obj.label] += 1
            
            # Count by category
            for category, labels in category_mappings.items():
                if obj.label in labels:
                    category_counts[category] += 1
                    break
                    
        # Determine the dominant category
        dominant_category = max(category_counts.items(), key=lambda x: x[1])
        
        # Apply classification rules
        if dominant_category[0] == "people" and dominant_category[1] >= 2:
            return "social"
            
        if dominant_category[0] == "vehicles" and dominant_category[1] >= 1:
            if "road" in object_types or "highway" in object_types:
                return "road"
            else:
                return "transportation"
                
        if dominant_category[0] == "furniture" and dominant_category[1] >= 1:
            if category_counts["kitchenware"] >= 1:
                return "kitchen"
            elif "bed" in object_types:
                return "bedroom"
            elif "couch" in object_types or "tv" in object_types:
                return "living_room"
            elif "dining table" in object_types:
                return "dining_room"
            else:
                return "indoor"
                
        if dominant_category[0] == "nature" and dominant_category[1] >= 1:
            if "tree" in object_types and object_types["tree"] >= 3:
                return "forest"
            elif "water" in object_types:
                return "coast"
            else:
                return "nature"
                
        if dominant_category[0] == "food" and dominant_category[1] >= 2:
            return "food"
            
        if dominant_category[0] == "animals" and dominant_category[1] >= 1:
            if category_counts["nature"] >= 1:
                return "wildlife"
            else:
                return "animal"
                
        if dominant_category[0] == "electronics" and category_counts["furniture"] >= 1:
            return "office"
            
        if dominant_category[0] == "sports" and dominant_category[1] >= 1:
            return "sports"
            
        # Default fallback based on object count
        if len(objects) == 0:
            return "empty"
        elif len(objects) <= 2:
            return "minimal"
        else:
            return "general"
    
    def _detect_activities(self, objects: List[VisualObject]) -> List[Dict[str, Any]]:
        """
        Detect potential activities in the scene based on object combinations.
        
        Args:
            objects: Detected objects
            
        Returns:
            List of potential activities
        """
        activities = []
        
        # Get object labels
        labels = [obj.label for obj in objects]
        label_count = {}
        for label in labels:
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
            
        # Define activity patterns
        activity_patterns = [
            {
                "name": "dining",
                "confidence": 0.8,
                "required": ["dining table"],
                "optional": ["chair", "fork", "knife", "spoon", "bowl", "cup", "wine glass", "person"]
            },
            {
                "name": "working",
                "confidence": 0.7,
                "required": ["laptop"],
                "optional": ["person", "chair", "desk", "book", "cup"]
            },
            {
                "name": "watching TV",
                "confidence": 0.7,
                "required": ["tv"],
                "optional": ["person", "couch", "remote"]
            },
            {
                "name": "cooking",
                "confidence": 0.7,
                "required": ["oven"],
                "optional": ["person", "microwave", "refrigerator", "sink", "bowl", "knife"]
            },
            {
                "name": "reading",
                "confidence": 0.6,
                "required": ["book"],
                "optional": ["person", "chair", "couch"]
            },
            {
                "name": "traveling",
                "confidence": 0.7,
                "required": ["car", "truck", "bus", "motorcycle", "train", "airplane"],  # Any one of these
                "optional": ["person", "suitcase", "backpack"]
            },
            {
                "name": "sports",
                "confidence": 0.7,
                "required": ["sports ball", "frisbee", "tennis racket", "baseball bat", "skateboard", "surfboard"],  # Any one of these
                "optional": ["person"]
            }
        ]
        
        # Check for each activity pattern
        for pattern in activity_patterns:
            # For required items, check if any of them is present (OR logic)
            required_present = False
            for required_item in pattern["required"]:
                if required_item in label_count:
                    required_present = True
                    break
                    
            if not required_present:
                continue
                
            # Count optional items
            optional_count = 0
            for optional_item in pattern["optional"]:
                if optional_item in label_count:
                    optional_count += 1
                    
            # Calculate confidence based on optional items
            optional_factor = min(1.0, optional_count / max(1, len(pattern["optional"])))
            confidence = pattern["confidence"] * (0.7 + 0.3 * optional_factor)
            
            # Add to detected activities
            activities.append({
                "activity": pattern["name"],
                "confidence": confidence
            })
            
        return activities
        
    def describe_scene(self, scene: VisualScene) -> str:
        """
        Generate a comprehensive description of a scene.
        
        Args:
            scene: The visual scene to describe
            
        Returns:
            Scene description
        """
        # Start with basic scene information
        scene_type = scene.global_attributes.get("scene_type", "general")
        object_count = scene.global_attributes.get("object_count", len(scene.objects))
        
        description = f"This appears to be a {scene_type} scene containing {object_count} objects. "
        
        # Mention primary objects
        primary_objects = scene.global_attributes.get("primary_objects", [])
        if primary_objects:
            description += f"The main elements are: {', '.join(primary_objects)}. "
            
        # Mention potential activities
        activities = scene.global_attributes.get("activities", [])
        if activities:
            # Filter by confidence and sort
            high_confidence_activities = [a for a in activities if a["confidence"] > 0.6]
            if high_confidence_activities:
                high_confidence_activities.sort(key=lambda x: x["confidence"], reverse=True)
                activity_names = [a["activity"] for a in high_confidence_activities[:2]]
                description += f"The scene suggests {' or '.join(activity_names)}. "
            
        # Describe spatial composition
        spatial_description = self._generate_spatial_description(scene)
        if spatial_description:
            description += spatial_description
            
        return description
        
    def _generate_spatial_description(self, scene: VisualScene) -> str:
        """
        Generate description of spatial relationships in the scene.
        
        Args:
            scene: The visual scene
            
        Returns:
            Spatial description
        """
        if not scene.objects:
            return ""
            
        # Find significant relationships
        significant_relations = []
        
        for obj_id, obj in scene.objects.items():
            for rel in obj.relationships:
                if rel["confidence"] > 0.7:  # Only high-confidence relationships
                    target_obj = scene.get_object_by_id(rel["target_id"])
                    if target_obj:
                        significant_relations.append(
                            (obj.label, rel["type"], target_obj.label, rel["confidence"])
                        )
                        
        # Sort by confidence
        significant_relations.sort(key=lambda x: x[3], reverse=True)
        
        # Generate description from top relations
        if significant_relations:
            relations_text = []
            
            for obj1, rel_type, obj2, _ in significant_relations[:5]:  # Top 5 relations
                if rel_type == "above":
                    relations_text.append(f"the {obj1} is above the {obj2}")
                elif rel_type == "below":
                    relations_text.append(f"the {obj1} is below the {obj2}")
                elif rel_type == "left_of":
                    relations_text.append(f"the {obj1} is to the left of the {obj2}")
                elif rel_type == "right_of":
                    relations_text.append(f"the {obj1} is to the right of the {obj2}")
                elif rel_type == "contains":
                    relations_text.append(f"the {obj1} contains the {obj2}")
                elif rel_type == "inside":
                    relations_text.append(f"the {obj1} is inside the {obj2}")
                elif rel_type == "near":
                    relations_text.append(f"the {obj1} is near the {obj2}")
                elif rel_type == "touching":
                    relations_text.append(f"the {obj1} is touching the {obj2}")
                elif rel_type == "horizontally_aligned_with":
                    relations_text.append(f"the {obj1} is horizontally aligned with the {obj2}")
                elif rel_type == "vertically_aligned_with":
                    relations_text.append(f"the {obj1} is vertically aligned with the {obj2}")
                elif rel_type == "centered_with":
                    relations_text.append(f"the {obj1} is centered with the {obj2}")
                    
            if relations_text:
                return "In terms of spatial arrangement, " + "; ".join(relations_text) + "."
                
        return ""