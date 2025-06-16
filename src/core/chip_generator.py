import base64
import uuid
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.core.face_detector import FaceDetection
from src.utils.config import get_config
from src.utils.file_handler import FileHandler
from src.utils.logger import get_facial_vision_logger, timing_decorator

logger = get_facial_vision_logger(__name__)


@dataclass
class ChipQuality:
    """Quality assessment for face chips"""

    is_valid: bool
    blur_score: float
    resolution_score: float
    brightness_score: float
    contrast_score: float
    quality_score: float
    reasons: List[str]


class ChipGenerator:
    """Generate face chips (cropped faces) with clustering support and quality checks"""

    def __init__(
        self,
        chip_size: Tuple[int, int] = None,
        padding_ratio: float = 0.2,
        jpeg_quality: int = None,
        enable_quality_checks: bool = True,
    ):
        """
        Initialize chip generator

        Args:
            chip_size: Output size (width, height). Default from config
            padding_ratio: Padding around face as ratio of face size (0.2 = 20%)
            jpeg_quality: JPEG compression quality (1-100). Default from config
            enable_quality_checks: Whether to perform quality assessments
        """
        config = get_config()
        output_config = config.get_output_config()

        self.chip_size = chip_size or tuple(output_config.get("chip_size", [224, 224]))
        self.padding_ratio = padding_ratio
        self.jpeg_quality = jpeg_quality or output_config.get("jpeg_quality", 85)
        self.use_base64 = output_config.get("use_base64", True)
        self.enable_quality_checks = enable_quality_checks

        # Quality thresholds
        self.quality_thresholds = {
            "min_blur_score": 100.0,  # Laplacian variance threshold
            "min_resolution": 64,  # Minimum face size
            "min_brightness": 20,  # Minimum average brightness
            "max_brightness": 235,  # Maximum average brightness
            "min_contrast": 10,  # Minimum standard deviation
            "min_quality_score": 0.5,  # Overall quality threshold
        }

        self.file_handler = FileHandler()

        logger.info(
            f"Initialized ChipGenerator",
            chip_size=self.chip_size,
            padding=self.padding_ratio,
            quality=self.jpeg_quality,
            quality_checks=self.enable_quality_checks,
        )

    @timing_decorator
    def generate_chip(
        self,
        image: Union[np.ndarray, str, Path],
        face_bbox: Tuple[int, int, int, int],
        output_path: Optional[Union[str, Path]] = None,
        return_base64: Optional[bool] = None,
        cluster_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate face chip from image and bounding box

        Args:
            image: Input image (numpy array or path)
            face_bbox: Face bounding box (top, right, bottom, left)
            output_path: Optional path to save the chip
            return_base64: Override default base64 setting
            cluster_id: Optional cluster ID for organized storage

        Returns:
            Dictionary with chip data and metadata
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image from {image}")

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        chip_name = f"face_chip_{timestamp}_{unique_id}"

        # Extract and process face chip
        face_chip = self._extract_face_with_padding(image, face_bbox)

        # Perform quality assessment
        quality = None
        if self.enable_quality_checks:
            quality = self.assess_quality(face_chip)
            if not quality.is_valid:
                logger.warning(
                    f"Low quality chip detected", chip_name=chip_name, reasons=quality.reasons
                )

        # Resize to standard size
        face_chip = self._resize_with_aspect_ratio(face_chip, self.chip_size)

        # Prepare result
        result = {
            "name": chip_name,
            "original_bbox": face_bbox,
            "chip_size": self.chip_size,
            "timestamp": datetime.now().isoformat(),
            "cluster_id": cluster_id,
            "quality": quality.__dict__ if quality else None,
        }

        # Handle clustered output path
        if output_path and cluster_id:
            output_path = self._get_clustered_path(output_path, cluster_id, chip_name)
        elif output_path:
            output_path = Path(output_path)
            if output_path.suffix == "":
                output_path = output_path.with_suffix(".jpg")

        # Handle output
        use_base64 = return_base64 if return_base64 is not None else self.use_base64

        if output_path:
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to file
            cv2.imwrite(str(output_path), face_chip, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            result["file_path"] = str(output_path)
            logger.debug(f"Saved face chip", path=str(output_path), cluster=cluster_id)

        if use_base64 or not output_path:
            # Convert to base64
            result["base64"] = self._encode_base64(face_chip)
            result["encoding"] = "base64"

        # Add chip array for further processing
        result["chip_array"] = face_chip

        return result

    def generate_clustered_chips(
        self,
        image: Union[np.ndarray, str, Path],
        face_detections: List[FaceDetection],
        cluster_ids: List[str],
        output_dir: Union[str, Path],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate chips organized by clusters

        Args:
            image: Source image
            face_detections: List of face detections
            cluster_ids: List of cluster IDs for each face
            output_dir: Base output directory

        Returns:
            Dictionary mapping cluster_id to list of chip data
        """
        if len(face_detections) != len(cluster_ids):
            raise ValueError("Number of detections must match number of cluster IDs")

        # Load image once
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image")

        output_dir = Path(output_dir)
        clustered_chips = {}

        for detection, cluster_id in zip(face_detections, cluster_ids):
            try:
                # Generate chip with cluster organization
                chip_data = self.generate_chip(
                    image=image,
                    face_bbox=detection.bbox,
                    output_path=output_dir,
                    cluster_id=cluster_id,
                )

                # Add detection metadata
                chip_data.update(
                    {
                        "confidence": detection.confidence,
                        "landmarks": detection.landmarks,
                        "face_area": detection.area,
                        "face_center": detection.center,
                    }
                )

                # Group by cluster
                if cluster_id not in clustered_chips:
                    clustered_chips[cluster_id] = []
                clustered_chips[cluster_id].append(chip_data)

            except Exception as e:
                logger.error(f"Error generating chip for cluster {cluster_id}", error=str(e))
                continue

        total_chips = sum(len(chips) for chips in clustered_chips.values())
        logger.info(f"Generated {total_chips} chips across {len(clustered_chips)} clusters")

        return clustered_chips

    def assess_quality(self, chip: np.ndarray) -> ChipQuality:
        """
        Assess the quality of a face chip

        Args:
            chip: Face chip as numpy array

        Returns:
            ChipQuality assessment
        """
        reasons = []

        # Convert to grayscale for some assessments
        if len(chip.shape) == 3:
            gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
        else:
            gray = chip

        # 1. Blur detection using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.quality_thresholds["min_blur_score"]:
            reasons.append(f"Blurry (score: {blur_score:.1f})")

        # 2. Resolution check
        height, width = gray.shape
        min_dimension = min(height, width)
        resolution_score = min_dimension / self.quality_thresholds["min_resolution"]
        if min_dimension < self.quality_thresholds["min_resolution"]:
            reasons.append(f"Low resolution ({min_dimension}px)")

        # 3. Brightness check
        brightness = np.mean(gray)
        brightness_score = 1.0
        if brightness < self.quality_thresholds["min_brightness"]:
            reasons.append(f"Too dark ({brightness:.1f})")
            brightness_score = brightness / self.quality_thresholds["min_brightness"]
        elif brightness > self.quality_thresholds["max_brightness"]:
            reasons.append(f"Too bright ({brightness:.1f})")
            brightness_score = self.quality_thresholds["max_brightness"] / brightness

        # 4. Contrast check
        contrast = np.std(gray)
        contrast_score = min(contrast / self.quality_thresholds["min_contrast"], 1.0)
        if contrast < self.quality_thresholds["min_contrast"]:
            reasons.append(f"Low contrast ({contrast:.1f})")

        # 5. Calculate overall quality score
        quality_score = np.mean(
            [
                min(blur_score / self.quality_thresholds["min_blur_score"], 1.0),
                resolution_score,
                brightness_score,
                contrast_score,
            ]
        )

        is_valid = (
            quality_score >= self.quality_thresholds["min_quality_score"] and len(reasons) == 0
        )

        return ChipQuality(
            is_valid=is_valid,
            blur_score=float(blur_score),
            resolution_score=float(resolution_score),
            brightness_score=float(brightness_score),
            contrast_score=float(contrast_score),
            quality_score=float(quality_score),
            reasons=reasons,
        )

    def _get_clustered_path(
        self, base_path: Union[str, Path], cluster_id: str, chip_name: str
    ) -> Path:
        """Get path for clustered chip storage"""
        base_path = Path(base_path)

        # Create cluster subdirectory
        cluster_dir = base_path / cluster_id

        # Generate filename with incrementing number
        chip_files = list(cluster_dir.glob("chip_*.jpg"))
        chip_number = len(chip_files) + 1

        filename = f"chip_{chip_number:03d}.jpg"
        return cluster_dir / filename

    def _extract_face_with_padding(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Extract face region with padding"""
        top, right, bottom, left = bbox
        height, width = image.shape[:2]

        # Calculate padding
        face_width = right - left
        face_height = bottom - top

        pad_x = int(face_width * self.padding_ratio)
        pad_y = int(face_height * self.padding_ratio)

        # Apply padding with bounds checking
        left_pad = max(0, left - pad_x)
        top_pad = max(0, top - pad_y)
        right_pad = min(width, right + pad_x)
        bottom_pad = min(height, bottom + pad_y)

        # Extract face region
        face_chip = image[top_pad:bottom_pad, left_pad:right_pad]

        if face_chip.size == 0:
            logger.warning(f"Empty face chip extracted from bbox {bbox}")
            # Return a small region as fallback
            face_chip = image[top:bottom, left:right]

        return face_chip

    def _resize_with_aspect_ratio(
        self, image: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Resize image maintaining aspect ratio"""
        height, width = image.shape[:2]
        target_width, target_height = target_size

        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)

        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create canvas and center the image
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # Calculate centering offsets
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        # Place resized image on canvas
        canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized

        return canvas

    def _encode_base64(self, image: np.ndarray) -> str:
        """Encode image to base64 string"""
        # Convert to RGB (PIL expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Save to bytes buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=self.jpeg_quality)

        # Encode to base64
        image_bytes = buffer.getvalue()
        base64_string = base64.b64encode(image_bytes).decode("utf-8")

        return base64_string

    def batch_generate(
        self,
        image: Union[np.ndarray, str, Path],
        face_bboxes: list,
        output_dir: Optional[Union[str, Path]] = None,
        cluster_ids: Optional[List[str]] = None,
    ) -> list:
        """
        Generate chips for multiple faces in an image

        Args:
            image: Input image
            face_bboxes: List of face bounding boxes
            output_dir: Directory to save chips
            cluster_ids: Optional cluster IDs for each face

        Returns:
            List of chip dictionaries
        """
        # Load image once
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image")

        # Create output directory if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Validate cluster IDs if provided
        if cluster_ids and len(cluster_ids) != len(face_bboxes):
            raise ValueError("Number of cluster IDs must match number of face bounding boxes")

        chips = []
        valid_chips = 0

        for idx, bbox in enumerate(face_bboxes):
            try:
                cluster_id = cluster_ids[idx] if cluster_ids else None

                chip_data = self.generate_chip(
                    image=image, face_bbox=bbox, output_path=output_dir, cluster_id=cluster_id
                )

                chips.append(chip_data)

                # Count valid chips
                if not self.enable_quality_checks or chip_data.get("quality", {}).get(
                    "is_valid", True
                ):
                    valid_chips += 1

            except Exception as e:
                logger.error(f"Error generating chip for face {idx}", error=str(e))
                continue

        logger.info(
            f"Generated {len(chips)} face chips from {len(face_bboxes)} detections",
            valid_chips=valid_chips,
        )
        return chips

    def filter_valid_chips(self, chips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter chips based on quality assessment

        Args:
            chips: List of chip dictionaries

        Returns:
            List of valid chips only
        """
        if not self.enable_quality_checks:
            return chips

        valid_chips = []
        for chip in chips:
            quality = chip.get("quality")
            if quality and quality.get("is_valid", True):
                valid_chips.append(chip)

        logger.info(f"Filtered {len(valid_chips)} valid chips from {len(chips)} total")
        return valid_chips

    def update_config(
        self,
        chip_size: Optional[Tuple[int, int]] = None,
        padding_ratio: Optional[float] = None,
        jpeg_quality: Optional[int] = None,
    ):
        """Update generator configuration"""
        if chip_size:
            self.chip_size = chip_size
        if padding_ratio is not None:
            self.padding_ratio = padding_ratio
        if jpeg_quality:
            self.jpeg_quality = jpeg_quality

        logger.info(
            f"Updated ChipGenerator config: size={self.chip_size}, "
            f"padding={self.padding_ratio}, quality={self.jpeg_quality}"
        )
