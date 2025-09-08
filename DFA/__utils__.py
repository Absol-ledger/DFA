

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_torch2_available() -> bool:
    """Check if PyTorch 2.0+ is available with scaled_dot_product_attention."""
    return hasattr(F, "scaled_dot_product_attention")


def get_device_info() -> Dict[str, Any]:
    """Get detailed device information."""
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        device_info['device_name'] = torch.cuda.get_device_name()
        device_info['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return device_info


def setup_memory_efficient_training(model: torch.nn.Module) -> None:
    """Setup memory efficient training configurations."""
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("Flash attention enabled")
        except:
            logger.warning("Flash attention not available")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
