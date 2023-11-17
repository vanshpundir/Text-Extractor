import subprocess


def extractText(image_path):
    # Define the command as a list of strings

    command = ["paddleocr", f"--image_dir={image_path}", "--type=structure", "--layout=True"]

    # Run the command
    try:
        subprocess.run(command, check=True, shell=False)
    except subprocess.CalledProcessError as e:
        print("An error occurred:", e)


