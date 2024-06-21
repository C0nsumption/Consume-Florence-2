# main.py

from analyze import ImageAnalyzer

# Example usage:
if __name__ == "__main__":
    model_id = './Florence-2-large'
    # image_source = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image_source = '/home/c0nsume/dev/florence/dataset/005.png'
    analyzer = ImageAnalyzer(model_id, image_source)

    # Running an example task
    result = analyzer('<CAPTION>')
    print(result)

    result = analyzer('<DETAILED_CAPTION>')
    print(result)

    result = analyzer('<MORE_DETAILED_CAPTION>')
    print(result)

    # # Running an object detection task
    od_result = analyzer('<OD>')
    analyzer.plot_bbox(od_result)

    # Running a dense region captioning task
    dense_region_result = analyzer('<DENSE_REGION_CAPTION>')
    analyzer.plot_bbox(dense_region_result)

    # Running a region proposal task
    region_proposal_result = analyzer('<REGION_PROPOSAL>')
    analyzer.plot_bbox(region_proposal_result)

    # Running a phrase grounding task
    phrase_grounding_result = analyzer('<CAPTION_TO_PHRASE_GROUNDING>', text_input="A woman")
    print(phrase_grounding_result)
    analyzer.plot_bbox(phrase_grounding_result)

    # Running a referring expression segmentation task
    referring_expression_result = analyzer('<REFERRING_EXPRESSION_SEGMENTATION>', text_input="a woman")
    analyzer.draw_polygons(referring_expression_result, fill_mask=True)

    # Running a region to segmentation task
    region_to_segmentation_result = analyzer('<REGION_TO_SEGMENTATION>', text_input="<loc_702><loc_575><loc_866><loc_772>")
    # print(region_to_segmentation_result)
    analyzer.draw_polygons(region_to_segmentation_result, fill_mask=True)

    # Running an open vocabulary detection task
    open_vocab_result = analyzer('<OPEN_VOCABULARY_DETECTION>', text_input="a green car")
    bbox_results = analyzer.convert_to_od_format(open_vocab_result)
    print(bbox_results)
    analyzer.plot_bbox(bbox_results)

    # Running a region to category task
    region_to_category_result = analyzer('<REGION_TO_CATEGORY>', text_input="<loc_52><loc_332><loc_932><loc_774>")

    # Running a region to description task
    region_to_description_result = analyzer('<REGION_TO_DESCRIPTION>', text_input="<loc_52><loc_332><loc_932><loc_774>")

    # OCR tasks
    url = "/home/c0nsume/dev/florence/assets/image.png"
    ocr_analyzer = ImageAnalyzer(model_id, url)

    # Running OCR task
    ocr_result = ocr_analyzer('<OCR>')

    # Running OCR with region task
    ocr_with_region_result = ocr_analyzer('<OCR_WITH_REGION>')
    ocr_analyzer.draw_ocr_bboxes(ocr_analyzer.image, ocr_with_region_result)
