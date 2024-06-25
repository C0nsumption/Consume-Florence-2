import os
import shutil
import sys

# Add the src directory to the Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_dir)

from image_analyzer import ImageAnalyzer

def main():
    # Get the absolute path of the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Ensure the output directory exists and is empty
    output_dir = os.path.join(project_root, "outputs")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Define the model ID and image paths using absolute paths
    model_id = os.path.join(project_root, 'Florence-2-large')
    image_source = os.path.join(project_root, 'dataset', '000.png')
    ocr_image_source = os.path.join(project_root, 'assets', 'image.png')

    # Create ImageAnalyzer instances
    analyzer = ImageAnalyzer(model_id, image_source)
    ocr_analyzer = ImageAnalyzer(model_id, ocr_image_source)

    # List of tasks to test
    tasks = [
        ("<CAPTION>", None),
        ("<DETAILED_CAPTION>", None),
        ("<MORE_DETAILED_CAPTION>", None),
        ("<OD>", None),
        ("<DENSE_REGION_CAPTION>", None),
        ("<REGION_PROPOSAL>", None),
        ("<CAPTION_TO_PHRASE_GROUNDING>", "A woman standing on the beach underneath a pier"),
        ("<REFERRING_EXPRESSION_SEGMENTATION>", "a woman"),
        ("<REGION_TO_SEGMENTATION>", "<loc_702><loc_575><loc_866><loc_772>"),
        ("<OPEN_VOCABULARY_DETECTION>", "a woman"),
        ("<REGION_TO_CATEGORY>", "<loc_52><loc_332><loc_932><loc_774>"),
        ("<REGION_TO_DESCRIPTION>", "<loc_52><loc_332><loc_932><loc_774>"),
        ("<OCR>", None),
        ("<OCR_WITH_REGION>", None)
    ]

    # Run tests for each task
    for task, text_input in tasks:
        print(f"\nRunning task: {task}")
        if task in ["<OCR>", "<OCR_WITH_REGION>"]:
            result = ocr_analyzer(task, text_input)
        else:
            result = analyzer(task, text_input)
        print(f"Result: {result}")

        # Handle specific post-processing for certain tasks
        if task in ["<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>", "<CAPTION_TO_PHRASE_GROUNDING>"]:
            analyzer.plot_bbox(result)
        elif task == "<REFERRING_EXPRESSION_SEGMENTATION>":
            analyzer.draw_polygons(result, fill_mask=True)
        elif task == "<REGION_TO_SEGMENTATION>":
            analyzer.draw_polygons(result, fill_mask=True)
        elif task == "<OPEN_VOCABULARY_DETECTION>":
            bbox_results = analyzer.convert_to_od_format(result)
            analyzer.plot_bbox(bbox_results)
        elif task == "<OCR_WITH_REGION>":
            ocr_analyzer.draw_ocr_bboxes(ocr_analyzer.image, result)

    print("\nAll tests completed. Please check the 'outputs' directory for results.")

if __name__ == "__main__":
    main()