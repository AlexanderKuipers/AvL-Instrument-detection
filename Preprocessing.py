import os
import shutil
import cv2
import random

# Emptying targetdirectory
directory_paths = [
    r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\CholecAHK\train\labels',
    r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\CholecAHK\train\images',
    r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\CholecAHK\test\labels',
    r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\CholecAHK\test\images',
    r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\CholecAHK\val\labels',
    r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\CholecAHK\val\images'

]

print(f"Start of emptying Directory paths")

# Iterate through the directory paths and empty each directory
for directory_path in directory_paths:
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # List all files and subdirectories in the directory
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                # If it's a file, delete it
                os.remove(item_path)
            elif os.path.isdir(item_path):
                # If it's a subdirectory, delete it recursively
                shutil.rmtree(item_path)
    else:
        print(f"Directory does not exist: {directory_path}")

print(f"Directory paths are emptied")


### Train data
## Preprocessing train annotation. Reducing the annotations by extracting every 25th file. The names of the annotations and images (.jpg) are identical

print(f"Start preprocessing train label data")

train_label_input_directory = r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\labels_original_train'
train_label_output_directory = r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\CholecAHK\train\labels'

if not os.path.exists(train_label_output_directory):
    os.makedirs(train_label_output_directory)

# Function to check if a filename should be kept
def should_keep(filename):
    # Get the part before the extension (if any)
    filename_without_extension = os.path.splitext(filename)[0]

    # Check if the filename starts with "frame_" and the part before the extension is a multiple of 25
    return filename_without_extension.startswith("frame_") and int(filename_without_extension.split("_")[1]) % 25 == 0

# Recursively process all subdirectories within the main directory
for root, _, files in os.walk(train_label_input_directory):
    for filename in files:
        if should_keep(filename) and filename.endswith(".txt"):
            source_file = os.path.join(root, filename)

            # Extract the video name from the directory structure
            video_name = os.path.basename(os.path.dirname(root))
            # Extract the frame number from the filename, such as "000000" for "frame_000000.txt"
            frame_number = filename.split("_")[1]
            # Construct the destination filename with "videoXanno_frame_000000"
            destination_filename = f"{video_name}_frame_{frame_number}"
            # Construct the full destination file path in the output directory
            destination_file = os.path.join(train_label_output_directory, destination_filename)
            # Copy the text file to the output directory with the modified name
            shutil.copy(source_file, destination_file)

print(f"Finished preprocessing train label data")

## Preprocessing train video. Reducing the videos by extracting every 25th image.

print(f"Start preprocessing train image data")
train_image_input_directory = r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\images_original_train'
train_image_output_directory = r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\CholecAHK\train\images'

# Interval to extract frames (every 25th frame)
interval = 25

# Iterate through the video files in the input directory
for filename in os.listdir(train_image_input_directory):
    video_file = os.path.join(train_image_input_directory, filename)
    video_number = os.path.splitext(filename)[0]  # Extract video number from the file name

    cap = cv2.VideoCapture(video_file)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % interval == 0:
            frame_filename = f"{video_number}anno_frame_{frame_index:06d}.jpg"
            output_path = os.path.join(train_image_output_directory, frame_filename)
            cv2.imwrite(output_path, frame)

        frame_index += 1

    cap.release()

print(f"Finished preprocessing train image data")

### Test data
## Preprocessing test annotation
print(f"Start preprocessing test label data")
test_label_input_directory = r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\labels_original_test'
test_label_output_directory = r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\CholecAHK\test\labels'

if not os.path.exists(test_label_output_directory):
    os.makedirs(test_label_output_directory)

if not os.path.exists(test_label_output_directory):
    os.makedirs(test_label_output_directory)

# Process all files in the input directory
for video_folder in os.listdir(test_label_input_directory):
    video_folder_path = os.path.join(test_label_input_directory, video_folder)
    if os.path.isdir(video_folder_path):
        obj_train_data_dir = os.path.join(video_folder_path, 'obj_train_data')
        if os.path.exists(obj_train_data_dir):
            for filename in os.listdir(obj_train_data_dir):
                if filename.endswith(".txt"):
                    source_file = os.path.join(obj_train_data_dir, filename)

                    # Construct the destination filename with the video name
                    destination_filename = f"{video_folder}_{filename}"
                    # Construct the full destination file path in the output directory
                    destination_file = os.path.join(test_label_output_directory, destination_filename)
                    # Copy the text file to the output directory with the modified name
                    shutil.copy(source_file, destination_file)

print(f"Finished preprocessing test label data")

## Preprocessing test video.

print(f"Start preprocessing test image data")

test_image_input_directory = r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\images_original_test'
test_image_output_directory = r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\CholecAHK\test\images'

if not os.path.exists(test_image_output_directory):
    os.makedirs(test_image_output_directory)

for filename in os.listdir(test_image_input_directory):
    video_file = os.path.join(test_image_input_directory, filename)
    video_number = os.path.splitext(filename)[0]  # Extract video number from the file name

    cap = cv2.VideoCapture(video_file)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = f"{video_number}_anno_frame_{frame_index:06d}.jpg"
        output_path = os.path.join(test_image_output_directory, frame_filename)
        cv2.imwrite(output_path, frame)

        frame_index += 1

    cap.release()

print(f"Finished preprocessing test image data")

### Validation data
## Split the train data into train-validation with a split of 80-20.
# Set the paths to the parent directories for images and labels
print(f"Start preprocesisng validation data")

# Set the paths to the parent directories for images and labels
image_parent_directory = r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\CholecAHK\train\images'
label_parent_directory = r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\CholecAHK\train\labels'

# Set the paths for the output validation directories for images and labels
val_image_output_directory = r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\CholecAHK\val\images'
val_label_output_directory = r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\CholecAHK\val\labels'

# Set the ratio for the train-validation split (e.g., 80% train, 20% validation)
split_ratio = 0.2  # Validation ratio

# Create the validation directories if they don't exist
if not os.path.exists(val_image_output_directory):
    os.makedirs(val_image_output_directory)
if not os.path.exists(val_label_output_directory):
    os.makedirs(val_label_output_directory)

# List all files in the image parent directory
files = os.listdir(image_parent_directory)

# Shuffle the files to randomize the selection
random.shuffle(files)

# Calculate the number of files for the validation set
num_files_validation = int(len(files) * split_ratio)

# Split the files into training and validation sets
validation_files = files[:num_files_validation]
train_files = files[num_files_validation:]

# Move the selected image and label files to the validation directories
for file in validation_files:
    # Move image file
    source_image_file = os.path.join(image_parent_directory, file)
    destination_image_file = os.path.join(val_image_output_directory, file)
    shutil.move(source_image_file, destination_image_file)

    # Move corresponding label file (assuming the same name with a .txt extension)
    label_file = os.path.splitext(file)[0] + '.txt'
    source_label_file = os.path.join(label_parent_directory, label_file)
    destination_label_file = os.path.join(val_label_output_directory, label_file)
    if os.path.exists(source_label_file):
        shutil.move(source_label_file, destination_label_file)

print("Data splitting and moving complete.")

