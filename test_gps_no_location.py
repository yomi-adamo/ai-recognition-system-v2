#!/usr/bin/env python3
"""Test GPS OCR extraction with 'no location data' messages"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.core.gps_ocr_extractor import GPSOCRExtractor

def create_test_frame_with_text(text: str, width: int = 1920, height: int = 1080) -> np.ndarray:
    """Create a test video frame with overlay text"""
    # Create white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a basic font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Add text to bottom right (typical overlay location)
    text_x = width - 400
    text_y = height - 100
    draw.text((text_x, text_y), text, fill='black', font=font)
    
    # Convert to numpy array
    return np.array(img)

def test_gps_extraction():
    """Test GPS extraction with different text scenarios"""
    extractor = GPSOCRExtractor()
    
    test_cases = [
        "2025-06-24 08:26:18 B8A44FBADA23 38.9549,-77.4120",  # Valid GPS
        "2025-06-24 08:26:18 B8A44FBADA23 NO LOCATION DATA",   # No location
        "2025-06-24 08:26:18 B8A44FBADA23 GPS DISABLED",       # GPS disabled
        "2025-06-24 08:26:18 B8A44FBADA23 Location Unavailable", # Unavailable
        "2025-06-24 08:26:18 B8A44FBADA23 No GPS Signal",      # No signal
        "2025-06-24 08:26:18 B8A44FBADA23",                    # No GPS info
    ]
    
    print("Testing GPS OCR extraction scenarios:\n")
    
    for i, text in enumerate(test_cases, 1):
        print(f"Test {i}: {text}")
        
        # Create test frame
        frame = create_test_frame_with_text(text)
        
        # Extract GPS
        result = extractor.extract_gps_from_frame(frame)
        
        if result:
            if result.get('status') == 'no_location_available':
                print(f"  Result: {result['status']} - {result['message']}")
            elif 'lat' in result and 'lon' in result:
                print(f"  Result: GPS coordinates found - lat={result['lat']}, lon={result['lon']}")
            else:
                print(f"  Result: Unexpected format - {result}")
        else:
            print("  Result: No GPS data found")
        
        print()

if __name__ == "__main__":
    test_gps_extraction()