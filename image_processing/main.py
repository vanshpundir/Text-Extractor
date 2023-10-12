from image_processing.image_orientation_handler import ImageOrientationHandler
from image_processing.image_orientation_checker import ImageOrientationChecker
from image_processing.vertical_to_horizontal import VerticalToHorizontal
from image_processing.crop_image import CropImage

def main():
    # Example usage of ImageOrientationHandler
    image_path = "/home/shivam/Documents/Github/Text-Extractor/images/cropped_image.jpg"
    handler = ImageOrientationHandler(image_path)
    handler.check_orientation()
    handler.rotate_and_save()

    # Example usage of ImageOrientationChecker
    image_path_checker = "/home/shivam/Documents/Github/Text-Extractor/images/cropped_image.jpg"
    checker = ImageOrientationChecker(image_path_checker)
    is_horizontal = checker.check_orientation()
    print("Is horizontal:", is_horizontal)

    # Example usage of VerticalToHorizontal
    image_path_vth = "/home/shivam/Documents/Github/Text-Extractor/images/cropped_image.jpg"
    converter = VerticalToHorizontal(image_path_vth)
    converter.save_as_jpeg('rotated.jpg')

    # Example usage of CropImage
    image_path_crop = "/home/shivam/Documents/Github/Text-Extractor/images/extract_data/IMG_9518.jpg"
    image_processor = CropImage(image_path_crop)
    image_processor.crop_image_to_largest_contour()
    image_processor.display_cropped_image()

if __name__ == '__main__':
    main()
