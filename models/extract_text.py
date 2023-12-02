import subprocess
import os

def extractText(image_path):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the Text-Extractor directory
    project_dir = os.path.dirname(script_dir)
    # Combine the script directory with the relative path to get the full path

    # Change the current working directory to Text-Extractor

    os.chdir(project_dir)

    # Define the command as a list of strings
    command = ["paddleocr", f"--image_dir={image_path}", "--type=structure", "--layout=True"]

    # Run the command
    try:
        subprocess.run(command, check=True, shell=False)
    except subprocess.CalledProcessError as e:
        print("An error occurred:", e)
#
if __name__ == "__main__":
    image_path = 'image_processing/rotated.jpg'  # Using a relative path for the image
    extractText(image_path)
