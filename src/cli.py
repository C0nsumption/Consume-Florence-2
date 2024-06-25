import argparse
import os
from image_analyzer import ImageAnalyzer  # Assuming your class is in a file named image_analyzer.py

def main():
    parser = argparse.ArgumentParser(description='Image Analyzer CLI')
    
    parser.add_argument('--model-id', type=str, required=True, help='Model ID to use for analysis')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the image file or URL')
    parser.add_argument('--task', type=str, required=True, choices=ImageAnalyzer.valid_tasks, help='Analysis task to perform')
    parser.add_argument('--text-input', type=str, help='Additional text input for the task, if required')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Create an instance of ImageAnalyzer
    analyzer = ImageAnalyzer(args.model_id, args.image_path)
    
    # Set the output directory
    analyzer.output_dir = args.output_dir
    
    # Perform the requested task
    result = analyzer(args.task, args.text_input)
    
    # Process the result based on the task
    if args.task in ['<OD>', '<OPEN_VOCABULARY_DETECTION>']:
        analyzer.plot_bbox(result)
    elif args.task in ['<CAPTION_TO_PHRASE_GROUNDING>', '<REFERRING_EXPRESSION_SEGMENTATION>', '<REGION_TO_SEGMENTATION>']:
        analyzer.draw_polygons(result)
    elif args.task in ['<OCR>', '<OCR_WITH_REGION>']:
        analyzer.draw_ocr_bboxes(analyzer.image, result)
    
    print(f"Analysis complete. Results saved in {args.output_dir}")

if __name__ == '__main__':
    main()