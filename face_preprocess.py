"""
Enhanced face preprocessing utilities with improved error handling and functionality.
"""

import cv2
import numpy as np
from skimage import transform as trans
from typing import Optional, Tuple, Union, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_image(img_path: Union[str, Path], **kwargs) -> Optional[np.ndarray]:
    """
    Read image with enhanced error handling and format validation.
    
    Args:
        img_path: Path to image file
        mode: 'rgb', 'bgr', or 'gray' 
        layout: 'HWC' or 'CHW'
        
    Returns:
        numpy array of image or None if failed
    """
    mode = kwargs.get('mode', 'rgb')
    layout = kwargs.get('layout', 'HWC')
    
    try:
        img_path = Path(img_path)
        if not img_path.exists():
            logger.error(f"Image file not found: {img_path}")
            return None
        
        if mode == 'gray':
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                logger.error(f"Failed to read image: {img_path}")
                return None
                
            if mode == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img is None:
            logger.error(f"Image loading returned None: {img_path}")
            return None
            
        if layout == 'CHW' and len(img.shape) == 3:
            img = np.transpose(img, (2, 0, 1))
            
        return img
        
    except Exception as e:
        logger.error(f"Error reading image {img_path}: {e}")
        return None


def validate_landmarks(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> bool:
    """
    Validate facial landmarks are within image boundaries and properly formatted.
    
    Args:
        landmarks: Array of shape (N, 2) containing landmark coordinates
        image_shape: (height, width) of the image
        
    Returns:
        True if landmarks are valid
    """
    if landmarks is None or len(landmarks) == 0:
        return False
        
    if landmarks.shape[1] != 2:
        logger.error(f"Landmarks should have shape (N, 2), got {landmarks.shape}")
        return False
    
    height, width = image_shape[:2]
    
    # Check if landmarks are within image boundaries with some tolerance
    if (landmarks[:, 0] < -10).any() or (landmarks[:, 0] > width + 10).any():
        logger.warning("Some landmarks are outside image width boundaries")
        return False
        
    if (landmarks[:, 1] < -10).any() or (landmarks[:, 1] > height + 10).any():
        logger.warning("Some landmarks are outside image height boundaries")
        return False
    
    return True


def preprocess(
    img: Union[str, np.ndarray], 
    bbox: Optional[np.ndarray] = None, 
    landmark: Optional[np.ndarray] = None, 
    **kwargs
) -> Optional[np.ndarray]:
    """
    Enhanced face preprocessing with better error handling and validation.
    
    Args:
        img: Input image (path or numpy array)
        bbox: Bounding box coordinates [x1, y1, x2, y2] 
        landmark: Facial landmarks array of shape (5, 2) for standard 5-point landmarks
        **kwargs: Additional parameters
            - image_size: Target size as 'H,W' string or tuple
            - margin: Margin for bbox expansion (default: 44)
            
    Returns:
        Preprocessed face image or None if failed
    """
    try:
        # Load image if path is provided
        if isinstance(img, (str, Path)):
            img = read_image(img, **kwargs)
            if img is None:
                return None
        
        # Validate image
        if img is None or len(img.shape) < 2:
            logger.error("Invalid image input")
            return None
        
        # Parse target image size
        image_size = parse_image_size(kwargs.get('image_size', ''))
        
        # Use landmark-based alignment if available
        if landmark is not None and validate_landmarks(landmark, img.shape):
            return align_face_with_landmarks(img, landmark, image_size)
        else:
            # Fallback to bbox-based cropping
            return crop_face_with_bbox(img, bbox, image_size, kwargs.get('margin', 44))
            
    except Exception as e:
        logger.error(f"Face preprocessing failed: {e}")
        return None


def parse_image_size(size_str: str) -> Optional[Tuple[int, int]]:
    """Parse image size from string format."""
    if not size_str:
        return None
        
    try:
        if ',' in size_str:
            parts = size_str.split(',')
            if len(parts) == 2:
                h, w = int(parts[0].strip()), int(parts[1].strip())
                if h > 0 and w > 0:
                    return (h, w)
        else:
            size = int(size_str.strip())
            if size > 0:
                return (size, size)
    except ValueError:
        pass
    
    logger.warning(f"Invalid image size format: {size_str}")
    return None


def align_face_with_landmarks(
    img: np.ndarray, 
    landmarks: np.ndarray, 
    target_size: Optional[Tuple[int, int]]
) -> Optional[np.ndarray]:
    """
    Align face using facial landmarks with robust transformation estimation.
    
    Args:
        img: Input image
        landmarks: Facial landmarks (5, 2)
        target_size: Target output size (height, width)
        
    Returns:
        Aligned face image
    """
    try:
        if target_size is None:
            target_size = (112, 112)
        
        # Standard reference landmarks for face alignment  
        # These are the canonical positions for eyes, nose, and mouth corners
        reference_landmarks = np.array([
            [30.2946, 51.6963],  # Left eye
            [65.5318, 51.5014],  # Right eye  
            [48.0252, 71.7366],  # Nose tip
            [33.5493, 92.3655],  # Left mouth corner
            [62.7299, 92.2041]   # Right mouth corner
        ], dtype=np.float32)
        
        # Adjust reference points if target width is 112
        if target_size[1] == 112:
            reference_landmarks[:, 0] += 8.0
        
        # Ensure landmarks are float32
        landmarks = landmarks.astype(np.float32)
        
        # Estimate similarity transformation
        tform = trans.SimilarityTransform()
        success = tform.estimate(landmarks, reference_landmarks)
        
        if not success:
            logger.warning("Failed to estimate transformation, using fallback method")
            return crop_face_with_bbox(img, None, target_size, 44)
        
        # Get transformation matrix
        M = tform.params[0:2, :]
        
        # Apply transformation
        aligned_face = cv2.warpAffine(
            img, M, (target_size[1], target_size[0]), 
            borderValue=0.0, 
            flags=cv2.INTER_LINEAR
        )
        
        return aligned_face
        
    except Exception as e:
        logger.error(f"Landmark-based alignment failed: {e}")
        return crop_face_with_bbox(img, None, target_size, 44)


def crop_face_with_bbox(
    img: np.ndarray,
    bbox: Optional[np.ndarray],
    target_size: Optional[Tuple[int, int]],
    margin: int = 44
) -> Optional[np.ndarray]:
    """
    Crop face using bounding box with margin expansion.
    
    Args:
        img: Input image
        bbox: Bounding box [x1, y1, x2, y2] or None for center crop
        target_size: Target output size (height, width)
        margin: Margin to expand bbox
        
    Returns:
        Cropped face image
    """
    try:
        height, width = img.shape[:2]
        
        if bbox is None:
            # Use center crop as fallback
            det = np.array([
                int(width * 0.0625),
                int(height * 0.0625), 
                width - int(width * 0.0625),
                height - int(height * 0.0625)
            ], dtype=np.int32)
        else:
            det = bbox.astype(np.int32)
        
        # Expand bbox with margin
        margin_half = margin // 2
        bb = np.array([
            max(det[0] - margin_half, 0),
            max(det[1] - margin_half, 0),
            min(det[2] + margin_half, width),
            min(det[3] + margin_half, height)
        ], dtype=np.int32)
        
        # Crop image
        cropped = img[bb[1]:bb[3], bb[0]:bb[2]]
        
        if cropped.size == 0:
            logger.error("Cropping resulted in empty image")
            return None
        
        # Resize if target size specified
        if target_size is not None:
            cropped = cv2.resize(
                cropped, 
                (target_size[1], target_size[0]), 
                interpolation=cv2.INTER_LINEAR
            )
        
        return cropped
        
    except Exception as e:
        logger.error(f"Bbox-based cropping failed: {e}")
        return None


def preprocess_3pt(
    img: Union[str, np.ndarray],
    bbox: Optional[np.ndarray] = None,
    landmark: Optional[np.ndarray] = None,
    **kwargs
) -> Optional[np.ndarray]:
    """
    Face preprocessing using 3-point landmarks (eyes and nose).
    
    This is a simplified version using only the first 3 landmarks
    for cases where full 5-point landmarks are not available.
    """
    try:
        if isinstance(img, (str, Path)):
            img = read_image(img, **kwargs)
            if img is None:
                return None
        
        image_size = parse_image_size(kwargs.get('image_size', ''))
        
        if landmark is not None and len(landmark) >= 3:
            # Use only first 3 points (left eye, right eye, nose)
            landmark_3pt = landmark[:3].astype(np.float32)
            
            # Reference 3-point landmarks
            reference_3pt = np.array([
                [30.2946, 51.6963],  # Left eye
                [65.5318, 51.5014],  # Right eye
                [48.0252, 71.7366],  # Nose tip
            ], dtype=np.float32)
            
            if image_size and image_size[1] == 112:
                reference_3pt[:, 0] += 8.0
            
            # Estimate transformation using 3 points
            tform = trans.SimilarityTransform()
            success = tform.estimate(landmark_3pt, reference_3pt)
            
            if success and image_size:
                M = tform.params[0:2, :]
                warped = cv2.warpAffine(
                    img, M, (image_size[1], image_size[0]), 
                    borderValue=0.0
                )
                return warped
        
        # Fallback to bbox-based processing
        return crop_face_with_bbox(img, bbox, image_size, kwargs.get('margin', 44))
        
    except Exception as e:
        logger.error(f"3-point preprocessing failed: {e}")
        return None


def compute_face_angle(landmarks: np.ndarray) -> float:
    """
    Compute face rotation angle from landmarks.
    
    Args:
        landmarks: Facial landmarks array
        
    Returns:
        Rotation angle in degrees
    """
    if len(landmarks) < 2:
        return 0.0
    
    try:
        # Use eye landmarks to compute angle
        left_eye = landmarks[0]
        right_eye = landmarks[1] 
        
        # Calculate angle between eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return angle
        
    except Exception as e:
        logger.error(f"Failed to compute face angle: {e}")
        return 0.0


def enhance_face_image(img: np.ndarray, enhance_factor: float = 1.2) -> np.ndarray:
    """
    Apply basic image enhancement to face image.
    
    Args:
        img: Input face image
        enhance_factor: Enhancement strength
        
    Returns:
        Enhanced image
    """
    try:
        # Convert to float for processing
        img_float = img.astype(np.float32) / 255.0
        
        # Apply gamma correction for brightness
        gamma = 1.0 / enhance_factor
        enhanced = np.power(img_float, gamma)
        
        # Slight contrast enhancement
        enhanced = np.clip(enhanced * 1.1, 0, 1)
        
        # Convert back to uint8
        enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}")
        return img