#!/usr/bin/env python3
"""
Simple script to create a sample PPM image for testing
"""

import math
import os

def create_sample_ppm(filename, width=512, height=512):
    """Create a colorful test PPM image"""
    
    with open(filename, 'w') as f:
        # PPM header
        f.write("P3\n")
        f.write(f"{width} {height}\n")
        f.write("255\n")
        
        # Generate pixel data
        for y in range(height):
            for x in range(width):
                # Normalized coordinates
                fx = x / width
                fy = y / height
                
                # Create interesting patterns
                # Red: horizontal gradient
                r = int(255 * fx)
                
                # Green: vertical gradient
                g = int(255 * fy)
                
                # Blue: circular pattern
                center_x = width / 2
                center_y = height / 2
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_dist = math.sqrt(center_x**2 + center_y**2)
                
                # Circular rings
                ring_factor = math.sin(dist / max_dist * 8 * math.pi)
                b = int(128 + 127 * ring_factor)
                
                # Add some checker pattern
                check_size = 32
                if ((x // check_size) + (y // check_size)) % 2:
                    r = min(255, int(r * 1.3))
                    g = min(255, int(g * 0.7))
                    b = min(255, int(b * 1.1))
                
                # Clamp values
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                
                f.write(f"{r} {g} {b} ")
            f.write("\n")

def main():
    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)
    
    # Create different sized test images
    create_sample_ppm("images/sample.ppm", 512, 512)
    create_sample_ppm("images/sample_small.ppm", 256, 256)
    create_sample_ppm("images/sample_large.ppm", 1024, 768)
    
    print("Sample PPM images created in images/ directory:")
    print("  - sample.ppm (512x512)")
    print("  - sample_small.ppm (256x256)")
    print("  - sample_large.ppm (1024x768)")

if __name__ == "__main__":
    main()