# analyze.py

import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import copy
import os
import glob

class ImageAnalyzer:
    colormap = [
        'blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray',
        'olive', 'cyan', 'red', 'lime', 'indigo', 'violet', 'aqua',
        'magenta', 'coral', 'gold', 'tan', 'skyblue'
    ]

    valid_tasks = {
        '<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>',
        '<OD>', '<DENSE_REGION_CAPTION>', '<REGION_PROPOSAL>',
        '<CAPTION_TO_PHRASE_GROUNDING>', '<REFERRING_EXPRESSION_SEGMENTATION>',
        '<REGION_TO_SEGMENTATION>', '<OPEN_VOCABULARY_DETECTION>',
        '<REGION_TO_CATEGORY>', '<REGION_TO_DESCRIPTION>', '<OCR>', '<OCR_WITH_REGION>'
    }

    def __init__(self, model_id, image_source):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model_id = model_id
        self.image = self.load_image(image_source)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.output_dir = 'outputs'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_input_image(image_source)

    def load_image(self, image_source):
        if os.path.exists(image_source):
            return Image.open(image_source).convert('RGB')
        elif image_source.startswith(('http://', 'https://')):
            return Image.open(requests.get(image_source, stream=True).raw).convert('RGB')
        else:
            raise ValueError(f"Invalid image source: {image_source}")

    def save_input_image(self, image_source):
        input_image_save_path = self.get_next_file_path(self.output_dir, 'png')
        if image_source.startswith(('http://', 'https://')):
            self.image.save(input_image_save_path)
        else:
            if os.path.exists(image_source):
                Image.open(image_source).convert('RGB').save(input_image_save_path)
                
    def get_next_file_path(self, directory, extension):
        pattern = os.path.join(directory, f'*{os.path.extsep}{extension}')
        files = glob.glob(pattern)
        if not files:
            return os.path.join(directory, f'000000{os.path.extsep}{extension}')
        latest_file = max(files, key=os.path.getctime)
        base_name = os.path.splitext(os.path.basename(latest_file))[0]
        next_number = int(base_name) + 1
        return os.path.join(directory, f'{next_number:06d}{os.path.extsep}{extension}')

    def get_output_path(self, task_name, extension='png'):
        task_dir = os.path.join(self.output_dir, task_name)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
        return self.get_next_file_path(task_dir, extension)
    
    def process_task(self, task_prompt, text_input=None):
        if task_prompt not in self.valid_tasks:
            raise ValueError(f"Invalid task: {task_prompt}. Available options: {', '.join(self.valid_tasks)}")

        prompt = task_prompt if text_input is None else task_prompt + text_input
        inputs = self.processor(text=prompt, images=self.image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(self.image.width, self.image.height)
        )
        # Save text output
        task_name = task_prompt.strip('<>').replace(' ', '_').lower()
        self.save_text_output(task_name, str(parsed_answer))


        return parsed_answer


    def plot_bbox(self, data):
        # Extract task name from the keys in the data dictionary
        task_name = list(data.keys())[0]
        dir_name = task_name.strip('<>').replace(' ', '_').lower()
        
        image_copy = copy.deepcopy(self.image)
        draw = ImageDraw.Draw(image_copy)
        font = ImageFont.load_default()

        for bbox, label in zip(data[task_name]['bboxes'], data[task_name]['labels']):
            x1, y1, x2, y2 = bbox
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            if label:
                draw.text((x1, y1), label, fill='white', font=font)

        # Save image output to the task-specific directory
        output_path = self.get_output_path(dir_name, 'png')
        image_copy.save(output_path)
        print(f"Image saved to {output_path}")

    def draw_polygons(self, prediction, fill_mask=False):
        # Extract task name from the keys in the prediction dictionary
        task_name = list(prediction.keys())[0]
        dir_name = task_name.strip('<>').replace(' ', '_').lower()

        image_copy = copy.deepcopy(self.image)
        draw = ImageDraw.Draw(image_copy)

        for polygons, label in zip(prediction[task_name]['polygons'], prediction[task_name]['labels']):
            color = random.choice(self.colormap)
            fill_color = color if fill_mask else None

            for _polygon in polygons:
                _polygon = np.array(_polygon).reshape(-1, 2)
                if len(_polygon) < 3:
                    print('Invalid polygon:', _polygon)
                    continue

                _polygon = _polygon.reshape(-1).tolist()
                draw.polygon(_polygon, outline=color, fill=fill_color)
                draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)

        # Save image output to the task-specific directory
        output_path = self.get_output_path(dir_name, 'png')
        image_copy.save(output_path)
        print(f"Image saved to {output_path}")
        # image_copy.show()




    def draw_ocr_bboxes(self, image, prediction):
        # Extract task name from the keys in the prediction dictionary
        task_name = list(prediction.keys())[0]
        dir_name = task_name.strip('<>').replace(' ', '_').lower()
        
        image_copy = copy.deepcopy(image)
        draw = ImageDraw.Draw(image_copy)
        font = ImageFont.load_default()

        bboxes, labels = prediction[task_name]['quad_boxes'], prediction[task_name]['labels']
        for box, label in zip(bboxes, labels):
            color = random.choice(self.colormap)
            new_box = np.array(box).tolist()
            draw.polygon(new_box, width=3, outline=color)
            draw.text((new_box[0]+8, new_box[1]+2), label, align="right", fill=color, font=font)

        # Save image output
        output_path = self.get_output_path(dir_name, 'png')
        image_copy.save(output_path)
        print(f"Image saved to {output_path}")


    def save_text_output(self, task_name, text_output):
        output_path = self.get_output_path(task_name, 'txt')
        with open(output_path, 'w') as f:
            f.write(text_output)
        print(f"Text output saved to {output_path}")



    def convert_to_od_format(self, data):
        task_name = list(data.keys())[0]
        task_data = data[task_name]
        
        # Extract bounding boxes and labels
        bboxes = task_data.get('bboxes', [])
        labels = task_data.get('bboxes_labels', [])
        
        # Construct the output format
        od_results = {
            task_name: {
                'bboxes': bboxes,
                'labels': labels
            }
        }
        
        return od_results



    def __call__(self, task_prompt, text_input=None):
        result = self.process_task(task_prompt, text_input)
        # print(result)
        return result
