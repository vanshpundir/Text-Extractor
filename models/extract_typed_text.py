import subprocess

# Define the command as a list of strings
command = ["paddleocr", "--image_dir=/content/col28.jpeg", "--type=structure", "--layout=True"]

# Run the command
try:
    subprocess.run(command, check=True, shell=False)
except subprocess.CalledProcessError as e:
    print("An error occurred:", e)
