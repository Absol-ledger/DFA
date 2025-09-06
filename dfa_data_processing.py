

import os
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import hashlib
import argparse
import random
import torch
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache setup for /tmp to avoid quota issues
USER = os.environ.get('USER', 'xz223')
TMP_CACHE_BASE = f'/tmp/{USER}/ai_model_cache'
os.makedirs(TMP_CACHE_BASE, exist_ok=True)

os.environ['HF_HOME'] = f'{TMP_CACHE_BASE}/huggingface'
os.environ['TRANSFORMERS_CACHE'] = f'{TMP_CACHE_BASE}/huggingface'
os.environ['TORCH_HOME'] = f'{TMP_CACHE_BASE}/torch'
os.environ['DEEPFACE_HOME'] = f'{TMP_CACHE_BASE}/deepface'

PROJECT_DIR = Path('/vol/bitbucket/xz223/dlenv/PPG')
import sys
sys.path.append(str(PROJECT_DIR))

from face_preprocess import preprocess
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    BLIP2_AVAILABLE = True
except ImportError:
    BLIP2_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False


class PromptComplexityClassifier:
    """Classify prompt complexity into simple/medium/complex categories"""
    
    def __init__(self):
        # Keywords indicating different complexity levels
        self.complex_keywords = {
            'artistic': ['cinematic', 'volumetric', 'dramatic lighting', 'studio lighting', 
                        'professional photography', 'award winning', 'masterpiece'],
            'technical': ['8k', '4k', 'ultra detailed', 'photorealistic', 'hyperrealistic',
                          'octane render', 'unreal engine', 'ray tracing'],
            'style': ['baroque', 'renaissance', 'impressionist', 'surreal', 'minimalist',
                     'abstract', 'contemporary', 'vintage'],
            'composition': ['rule of thirds', 'golden ratio', 'depth of field', 'bokeh',
                           'focal length', 'wide angle', 'macro']
        }
        
        self.medium_keywords = {
            'quality': ['high quality', 'detailed', 'sharp', 'clear', 'professional'],
            'lighting': ['natural lighting', 'soft lighting', 'ambient', 'well-lit'],
            'mood': ['happy', 'serious', 'calm', 'energetic', 'peaceful']
        }
        
        self.simple_keywords = ['person', 'portrait', 'photo', 'face', 'headshot']
    
    def classify(self, prompt: str, attributes: Dict = None) -> Tuple[str, Dict]:
        """
        Classify prompt complexity and return classification details
        
        Returns:
            complexity: 'simple', 'medium', or 'complex'
            details: Dictionary with classification metrics
        """
        if not prompt:
            return 'simple', {'word_count': 0, 'keyword_matches': {}}
        
        prompt_lower = prompt.lower()
        words = prompt_lower.split()
        word_count = len(words)
        
        # Count keyword matches
        complex_matches = 0
        medium_matches = 0
        matched_categories = []
        
        # Check complex keywords
        for category, keywords in self.complex_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    complex_matches += 1
                    matched_categories.append(f"complex_{category}")
        
        # Check medium keywords
        for category, keywords in self.medium_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    medium_matches += 1
                    matched_categories.append(f"medium_{category}")
        
        # Calculate additional metrics
        has_multiple_descriptors = len(re.findall(r'\b(with|wearing|showing|featuring)\b', prompt_lower)) >= 2
        has_technical_specs = bool(re.search(r'\b\d+k\b|\bf/\d+|\bmm\b', prompt_lower))
        has_artistic_terms = any(term in prompt_lower for term in ['composition', 'aesthetic', 'style'])
        
        # Determine complexity
        complexity_score = 0
        
        # Word count contribution
        if word_count > 15:
            complexity_score += 3
        elif word_count > 10:
            complexity_score += 2
        elif word_count > 5:
            complexity_score += 1
        
        # Keyword contribution
        complexity_score += complex_matches * 2
        complexity_score += medium_matches
        
        # Special features contribution
        if has_multiple_descriptors:
            complexity_score += 2
        if has_technical_specs:
            complexity_score += 2
        if has_artistic_terms:
            complexity_score += 1
        
        # Classify based on score
        if complexity_score >= 8:
            complexity = 'complex'
        elif complexity_score >= 4:
            complexity = 'medium'
        else:
            complexity = 'simple'
        
        # Override with attributes if highly detailed
        if attributes and len([v for v in attributes.values() if v and v != 'neutral' and v != 'person']) >= 4:
            if complexity == 'simple':
                complexity = 'medium'
        
        details = {
            'word_count': word_count,
            'complexity_score': complexity_score,
            'complex_keyword_matches': complex_matches,
            'medium_keyword_matches': medium_matches,
            'matched_categories': matched_categories,
            'has_multiple_descriptors': has_multiple_descriptors,
            'has_technical_specs': has_technical_specs,
            'has_artistic_terms': has_artistic_terms
        }
        
        return complexity, details


# Import previous helper functions
def detect(image, face_detection):
    """Face detection using ModelScope RetinaFace"""
    try:
        result_det = face_detection(image)
        if result_det is None or 'scores' not in result_det:
            return None
        confs = result_det['scores']
        idx = np.argmax(confs)
        pts = result_det['keypoints'][idx]
        points_vec = np.array(pts).reshape(5, 2)
        return points_vec
    except Exception as e:
        logger.debug(f"Face detection error: {e}")
        return None


def get_mask_head(result):
    """Generate head mask from segmentation result"""
    try:
        if not result or 'masks' not in result:
            return None
            
        masks = result['masks']
        scores = result['scores']
        labels = result['labels']
        
        img_shape = masks[0].shape
        mask_face = np.zeros(img_shape)
        mask_hair = np.zeros(img_shape)
        mask_human = np.zeros(img_shape)
        
        for i in range(len(labels)):
            if scores[i] > 0.8:
                if labels[i] == 'Face':
                    if np.sum(masks[i]) > np.sum(mask_face):
                        mask_face = masks[i]
                elif labels[i] == 'Human':
                    if np.sum(masks[i]) > np.sum(mask_human):
                        mask_human = masks[i]
                elif labels[i] == 'Hair':
                    if np.sum(masks[i]) > np.sum(mask_hair):
                        mask_hair = masks[i]
        
        mask_head = np.clip(mask_hair + mask_face, 0, 1)
        ksize = max(int(np.sqrt(np.sum(mask_face)) / 20), 1)
        kernel = np.ones((ksize, ksize))
        mask_head = cv2.dilate(mask_head, kernel, iterations=1) * mask_human
        
        _, mask_head = cv2.threshold((mask_head * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask_head, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
            
        areas = [cv2.contourArea(c) for c in contours]
        max_idx = np.argmax(areas)
        mask_head = np.zeros(img_shape).astype(np.uint8)
        cv2.fillPoly(mask_head, [contours[max_idx]], 255)
        mask_head = mask_head.astype(np.float32) / 255
        mask_head = np.clip(mask_head + mask_face, 0, 1)
        
        return np.expand_dims(mask_head, 2)
        
    except Exception as e:
        logger.debug(f"Mask generation error: {e}")
        return None


def align(image, points_vec):
    """Face alignment using 5-point landmarks"""
    try:
        if points_vec is None:
            return None
        warped = preprocess(np.array(image)[:,:,::-1], bbox=None, landmark=points_vec, image_size='112, 112')
        if warped is None:
            return None
        return Image.fromarray(warped[:,:,::-1])
    except Exception as e:
        logger.debug(f"Face alignment error: {e}")
        return None


class SmartCaptionGenerator:
    """AI-powered caption generator with complexity classification"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.complexity_classifier = PromptComplexityClassifier()
        self.init_models()
        
        # Templates organized by complexity
        self.templates_by_complexity = {
            'simple': [
                "a person",
                "a portrait",
                "a face photo",
                "a headshot"
            ],
            'medium': [
                "a professional portrait of a {gender}",
                "a {quality} headshot with {lighting}",
                "a {emotion} person in natural lighting",
                "a detailed portrait showing clear features"
            ],
            'complex': [
                "a professional studio portrait with dramatic lighting and sharp focus",
                "a cinematic headshot with volumetric lighting and photorealistic details",
                "an artistic portrait featuring professional photography techniques",
                "a high-quality portrait with studio lighting and bokeh background"
            ]
        }
    
    def init_models(self):
        """Initialize AI models"""
        logger.info("Initializing caption generation models...")
        
        self.blip2_processor = None
        self.blip2_model = None
        if BLIP2_AVAILABLE:
            try:
                model_name = "Salesforce/blip2-opt-2.7b"
                self.blip2_processor = Blip2Processor.from_pretrained(model_name)
                self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto" if torch.cuda.is_available() else "cpu"
                )
                logger.info("BLIP2 model loaded successfully")
            except Exception as e:
                logger.warning(f"BLIP2 loading failed: {e}")
        
        self.clip_model = None
        self.clip_preprocess = None
        if CLIP_AVAILABLE:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"CLIP loading failed: {e}")
        
        self.deepface_available = DEEPFACE_AVAILABLE
    
    def analyze_face_attributes(self, face_image: Image.Image) -> Dict[str, str]:
        """Analyze face attributes for caption generation"""
        attributes = {
            'age_group': 'person',
            'gender': 'person',
            'emotion': 'neutral',
            'lighting': 'natural lighting',
            'quality': 'clear'
        }
        
        if self.deepface_available:
            try:
                img_array = np.array(face_image)
                analysis = DeepFace.analyze(
                    img_array,
                    actions=['age', 'gender', 'emotion'],
                    enforce_detection=False
                )
                
                age = analysis[0].get('age', 30)
                attributes['age_group'] = 'young' if age < 30 else 'middle-aged' if age < 55 else 'elderly'
                
                gender = analysis[0].get('dominant_gender', 'person').lower()
                attributes['gender'] = gender if gender in ['man', 'woman'] else 'person'
                
                emotion = analysis[0].get('dominant_emotion', 'neutral').lower()
                attributes['emotion'] = 'smiling' if emotion in ['happy', 'joy'] else emotion
                
            except Exception as e:
                logger.debug(f"DeepFace analysis failed: {e}")
        
        # Image quality analysis
        img_array = np.array(face_image)
        brightness = np.mean(img_array)
        
        if brightness > 180:
            attributes['lighting'] = 'bright lighting'
        elif brightness < 80:
            attributes['lighting'] = 'soft lighting'
        
        # Sharpness analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var > 500:
            attributes['quality'] = 'sharp and detailed'
        elif laplacian_var < 100:
            attributes['quality'] = 'soft focus'
        
        return attributes
    
    def generate_caption_with_complexity(
        self,
        face_image: Image.Image,
        full_image: Optional[Image.Image] = None,
        target_complexity: Optional[str] = None
    ) -> Dict[str, any]:
        """Generate caption with automatic complexity classification"""
        
        # Analyze face attributes
        attributes = self.analyze_face_attributes(face_image)
        
        # Generate base caption
        base_caption = None
        if self.blip2_model and full_image:
            try:
                inputs = self.blip2_processor(full_image, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = self.blip2_model.generate(
                        **inputs,
                        max_new_tokens=30 if target_complexity == 'complex' else 20,
                        do_sample=True,
                        temperature=0.7,
                        num_beams=3
                    )
                
                base_caption = self.blip2_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()
                
            except Exception as e:
                logger.debug(f"BLIP2 generation failed: {e}")
        
        # Fallback to template if needed
        if not base_caption:
            complexity_level = target_complexity or 'medium'
            template = random.choice(self.templates_by_complexity[complexity_level])
            base_caption = template.format(**attributes)
        
        # Enhance caption based on target complexity
        if target_complexity == 'complex' and 'detailed' not in base_caption.lower():
            base_caption = f"highly detailed {base_caption} with professional photography"
        elif target_complexity == 'simple' and len(base_caption.split()) > 8:
            # Simplify overly complex captions
            words = base_caption.split()[:8]
            base_caption = ' '.join(words)
        
        # Classify the generated caption
        complexity, complexity_details = self.complexity_classifier.classify(base_caption, attributes)
        
        # Generate variants at different complexity levels
        variants = {
            'simple': self._simplify_caption(base_caption),
            'medium': self._moderate_caption(base_caption, attributes),
            'complex': self._complexify_caption(base_caption, attributes)
        }
        
        return {
            'prompt': base_caption,
            'complexity': complexity,
            'complexity_details': complexity_details,
            'variants': variants,
            'attributes': attributes,
            'negative_prompt': self._generate_negative_prompt(complexity)
        }
    
    def _simplify_caption(self, caption: str) -> str:
        """Create a simple version of the caption"""
        # Remove complex descriptors
        simple = caption.lower()
        remove_terms = ['highly detailed', 'professional', 'cinematic', 'dramatic', 'volumetric']
        for term in remove_terms:
            simple = simple.replace(term, '')
        
        # Clean up and limit words
        words = simple.split()[:5]
        simple = ' '.join(words).strip()
        
        if not simple or len(simple) < 5:
            simple = "a portrait photo"
        
        return simple
    
    def _moderate_caption(self, caption: str, attributes: Dict) -> str:
        """Create a medium complexity caption"""
        if len(caption.split()) < 8:
            # Add some attributes
            additions = []
            if attributes.get('quality') and 'quality' not in caption:
                additions.append(attributes['quality'])
            if attributes.get('lighting') and 'lighting' not in caption:
                additions.append(f"with {attributes['lighting']}")
            
            if additions:
                caption = f"{caption} {' '.join(additions)}"
        
        return caption
    
    def _complexify_caption(self, caption: str, attributes: Dict) -> str:
        """Create a complex version of the caption"""
        enhancements = []
        
        if 'professional' not in caption.lower():
            enhancements.append('professional')
        if 'detailed' not in caption.lower():
            enhancements.append('highly detailed')
        
        # Add technical specs
        technical = random.choice(['8k resolution', 'photorealistic', 'studio quality'])
        if technical not in caption.lower():
            enhancements.append(technical)
        
        # Add artistic elements
        artistic = random.choice(['dramatic lighting', 'cinematic composition', 'artistic photography'])
        if artistic not in caption.lower():
            enhancements.append(artistic)
        
        if enhancements:
            caption = f"{' '.join(enhancements)} {caption}"
        
        return caption
    
    def _generate_negative_prompt(self, complexity: str) -> str:
        """Generate negative prompt based on complexity level"""
        base_negative = ["low quality", "blurry", "distorted"]
        
        if complexity == 'complex':
            # More detailed negatives for complex prompts
            base_negative.extend([
                "amateur", "poorly lit", "out of focus",
                "bad anatomy", "ugly", "duplicate",
                "mutated", "bad proportions", "disfigured"
            ])
        elif complexity == 'medium':
            base_negative.extend([
                "bad quality", "unclear", "poorly drawn"
            ])
        
        return ", ".join(base_negative)


class EnhancedFACTDatasetBuilder:
    """Dataset builder with prompt complexity classification"""
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        quality_threshold: float = 0.5,
        max_samples: Optional[int] = None,
        device: str = 'cuda'
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.quality_threshold = quality_threshold
        self.max_samples = max_samples
        self.device = device
        
        # Create output directories
        self.dirs = {
            'images': self.output_dir / 'images',
            'faces': self.output_dir / 'faces',
            'masks': self.output_dir / 'masks',
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.setup_processors()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'complexity_distribution': {'simple': 0, 'medium': 0, 'complex': 0}
        }
        
        self.metadata = {
            'dataset_info': {
                'image_size': 512,
                'face_size': 112,
                'mask_size': 64,
                'processing_method': 'modelscope_with_complexity_classification'
            },
            'images': [],
            'identity_mapping': {}
        }
    
    def setup_processors(self):
        """Initialize processors"""
        logger.info("Initializing processors...")
        
        self.face_detection = pipeline(
            task=Tasks.face_detection,
            model='damo/cv_resnet50_face-detection_retinaface'
        )
        
        self.segmentation_pipeline = pipeline(
            task=Tasks.image_segmentation,
            model='damo/cv_resnet101_image-multiple-human-parsing'
        )
        
        self.caption_generator = SmartCaptionGenerator(device=self.device)
        
        logger.info("All processors initialized")
    
    def face_image_preprocess(self, image):
        """Preprocess face image"""
        try:
            result = self.segmentation_pipeline(image)
            if not result:
                return None
            
            mask_head = get_mask_head(result)
            if mask_head is None:
                return None
            
            image_masked = Image.fromarray((np.array(image) * mask_head).astype(np.uint8))
            
            points_vec = detect(image_masked, self.face_detection)
            if points_vec is None:
                return None
            
            aligned_face = align(image_masked, points_vec)
            if aligned_face is None:
                return None
            
            return {
                'aligned_face': aligned_face,
                'head_mask': mask_head,
                'landmarks': points_vec
            }
            
        except Exception as e:
            logger.debug(f"Preprocessing failed: {e}")
            return None
    
    def process_single_image(self, img_path: Path) -> Optional[Dict]:
        """Process single image with complexity classification"""
        self.stats['total_processed'] += 1
        
        try:
            # Load and resize image
            image = Image.open(img_path).convert('RGB')
            image_resized = image.resize((512, 512), Image.LANCZOS)
            
            # Face preprocessing
            result = self.face_image_preprocess(image_resized)
            if not result:
                self.stats['failed'] += 1
                return None
            
            # Generate filename
            img_hash = hashlib.md5(np.array(image_resized).tobytes()).hexdigest()[:8]
            identity_id = self._extract_identity_id(img_path)
            filename = f"{identity_id}_{img_hash}.jpg"
            
            # Save processed images
            image_resized.save(self.dirs['images'] / filename, quality=95)
            result['aligned_face'].save(self.dirs['faces'] / filename, quality=95)
            
            # Save mask
            training_mask = cv2.resize(
                result['head_mask'].squeeze(),
                (64, 64),
                interpolation=cv2.INTER_NEAREST
            )
            mask_img = Image.fromarray((training_mask * 255).astype(np.uint8))
            mask_img.save(self.dirs['masks'] / filename.replace('.jpg', '.png'))
            
            # Generate caption with complexity classification
            caption_result = self.caption_generator.generate_caption_with_complexity(
                face_image=result['aligned_face'],
                full_image=image_resized
            )
            
            # Update statistics
            complexity = caption_result['complexity']
            self.stats['complexity_distribution'][complexity] += 1
            self.stats['successful'] += 1
            
            return {
                'filename': filename,
                'identity_id': identity_id,
                'prompt': caption_result['prompt'],
                'complexity': complexity,
                'complexity_score': caption_result['complexity_details']['complexity_score'],
                'variants': caption_result['variants'],
                'attributes': caption_result['attributes'],
                'negative_prompt': caption_result['negative_prompt'],
                'complexity_details': caption_result['complexity_details']
            }
            
        except Exception as e:
            logger.debug(f"Processing failed for {img_path}: {e}")
            self.stats['failed'] += 1
            return None
    
    def _extract_identity_id(self, img_path: Path) -> str:
        """Extract identity ID from image path"""
        if img_path.parent != self.source_dir:
            return img_path.parent.name
        
        filename = img_path.stem
        if 'celeba' in str(self.source_dir).lower():
            try:
                file_id = int(filename.split('.')[0])
                return f"celeb_{file_id // 100:04d}"
            except:
                pass
        
        return filename[:10] if len(filename) > 10 else filename
    
    def process_dataset(self):
        """Process entire dataset"""
        # Collect image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(self.source_dir.rglob(f'*{ext}'))
        
        if self.max_samples:
            image_files = image_files[:self.max_samples]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images
        enhanced_prompts = {}
        
        for img_path in tqdm(image_files, desc="Processing images"):
            result = self.process_single_image(img_path)
            
            if result:
                self.metadata['images'].append(result)
                
                # Group by identity
                identity_id = result['identity_id']
                if identity_id not in self.metadata['identity_mapping']:
                    self.metadata['identity_mapping'][identity_id] = []
                self.metadata['identity_mapping'][identity_id].append(result['filename'])
                
                # Save enhanced prompts with complexity
                enhanced_prompts[result['filename']] = {
                    'prompt': result['prompt'],
                    'complexity': result['complexity'],
                    'complexity_score': result['complexity_score'],
                    'variants': result['variants'],
                    'attributes': result['attributes'],
                    'negative_prompt': result['negative_prompt'],
                    'complexity_details': result['complexity_details']
                }
        
        # Save all data
        self.save_metadata()
        self.save_enhanced_prompts(enhanced_prompts)
        self.print_statistics()
    
    def save_metadata(self):
        """Save metadata"""
        self.metadata['dataset_stats'] = {
            'total_images': len(self.metadata['images']),
            'total_identities': len(self.metadata['identity_mapping']),
            'processing_stats': self.stats
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_path}")
    
    def save_enhanced_prompts(self, enhanced_prompts: Dict):
        """Save enhanced prompts with complexity labels"""
        # Full version with all details
        prompts_path = self.output_dir / 'enhanced_prompts.json'
        with open(prompts_path, 'w') as f:
            json.dump(enhanced_prompts, f, indent=2)
        
        # Simplified version for training
        simple_prompts = {}
        for filename, data in enhanced_prompts.items():
            simple_prompts[filename] = {
                'prompt': data['prompt'],
                'negative_prompt': data['negative_prompt'],
                'complexity': data['complexity']  # Key addition: complexity label
            }
        
        simple_path = self.output_dir / 'prompts.json'
        with open(simple_path, 'w') as f:
            json.dump(simple_prompts, f, indent=2)
        
        logger.info(f"Enhanced prompts saved to: {prompts_path}")
        logger.info(f"Training prompts saved to: {simple_path}")
    
    def print_statistics(self):
        """Print processing statistics"""
        logger.info("=" * 70)
        logger.info("Dataset Processing Complete")
        logger.info("=" * 70)
        logger.info(f"Total processed: {self.stats['total_processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        
        logger.info("\nComplexity Distribution:")
        total = sum(self.stats['complexity_distribution'].values())
        if total > 0:
            for complexity, count in self.stats['complexity_distribution'].items():
                percentage = (count / total) * 100
                logger.info(f"  {complexity}: {count} ({percentage:.1f}%)")
        
        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='FACT Dataset Builder with Complexity Classification')
    parser.add_argument('--source', type=str, required=True, help='Source image directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--quality-threshold', type=float, default=0.5)
    parser.add_argument('--max-samples', type=int, help='Maximum samples for testing')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    builder = EnhancedFACTDatasetBuilder(
        source_dir=args.source,
        output_dir=args.output,
        quality_threshold=args.quality_threshold,
        max_samples=args.max_samples,
        device=args.device
    )
    
    builder.process_dataset()
    


if __name__ == '__main__':
    main()