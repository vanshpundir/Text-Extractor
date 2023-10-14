import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res
from PIL import Image


table_engine = PPStructure(show_log=True)

save_folder = '/Users/vansh/PycharmProjects/Text-Extractor/script/output'
img_path = '/Users/vansh/PycharmProjects/Text-Extractor/image_processing/rotated.jpg'
img = cv2.imread(img_path)
result = table_engine(img)
type(result)
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)



