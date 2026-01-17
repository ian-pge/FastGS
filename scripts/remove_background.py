import argparse
import os
from pathlib import Path
from rembg import remove
from PIL import Image
from tqdm import tqdm
import torch

def process_images(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    extensions = {'.png', '.jpg', '.jpeg', '.JPG', '.PNG'}
    image_files = [f for f in input_path.iterdir() if f.suffix in extensions]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Processing images with rembg to {output_dir}...")
    
    # Process images
    for image_file in tqdm(image_files):
        try:
            input_image = Image.open(image_file)
            
            # Apply background removal
            output_image = remove(input_image)
            
            # Save the result
            # Ensure we save as PNG to keep transparency
            output_filename = output_path / (image_file.stem + ".png")
            output_image.save(output_filename)
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove background from images using AI")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory (default: input_dir_masked)")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = str(Path(args.input).parent / (Path(args.input).name + "_masked"))
        
    process_images(args.input, args.output)
