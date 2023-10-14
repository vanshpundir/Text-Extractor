import os
import cv2
from paddleocr import PPStructure, save_structure_res

class Extract_Boundary_box:
    def __init__(self, img_path, save_folder='script/output'):
        self.img_path = img_path
        self.save_folder = save_folder
        self.table_engine = PPStructure(show_log=True)

    def extract_and_save_table(self):
        img = cv2.imread(self.img_path)
        result = self.table_engine(img)
        save_structure_res(result, self.save_folder, os.path.basename(self.img_path).split('.')[0])

        cleaned_result = []
        for line in result:
            line.pop('img')
            cleaned_result.append(line)

        return cleaned_result



if __name__ == "__main__":
    img_path = 'image_processing/rotated.jpg'
    table_extractor = Extract_Boundary_box(img_path)
    extracted_data = table_extractor.extract_and_save_table()
    print(extracted_data)
    for line in extracted_data:
        print(line)
