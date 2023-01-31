import requests
import shutil
import os
from PIL import Image
import json
from sklearn.model_selection import train_test_split

# Query URL
BASE_URL ='http://155.138.233.78:8000/api/'
# BASE_URL = 'http://localhost:8000/api/'
USER = 'pjohnson@example.com'
PASSWORD = 'MOlita01!'
LOGIN_ENDPOINT = 'users/login/token/'
TRAINING_ENDPOINT = 'trainings'


CROP_SHAPE = (512, 512)
SCALE = 10
DATASET_PATH = f"finetunning-dataset-{SCALE}-{CROP_SHAPE[0]}x{CROP_SHAPE[1]}"
WITH_ERRORS_PATH = os.path.join(DATASET_PATH, "with_errors")
WITHOUT_ERRORS_PATH = os.path.join(DATASET_PATH, "without_errors")
METADATA_PATH = os.path.join(DATASET_PATH, "metadata")


def authenticate():
    url = (f'{BASE_URL}{LOGIN_ENDPOINT}')
    res = requests.post(url,
                        headers={'accept': 'application/json',
                                 'Content-Type': 'application/x-www-form-urlencoded'},
                        data={'username': USER, 'password': PASSWORD})
    access_token = res.json()['access_token']
    headers = {'Authorization': f'Bearer {access_token}',
               'accept': 'application/json'}
    return headers


def get_trainings_info(scale):
    headers = authenticate()
    url = (f'{BASE_URL}{TRAINING_ENDPOINT}')
    res = requests.get(url, headers=headers, params={
                       "scale": scale, "show_discarded": False, "get_findings": True})
    trainings = res.json()
    return trainings


def download_image(url, name):
    page = requests.get(url)
    f_ext = os.path.splitext(url)[-1]
    with open(name, 'wb') as f:
        f.write(page.content)


def preprocess_data(trainings, crop_size):
    images = []
    findings_map = {0: [], 1: [], 2: []}
    num_trainings = len(trainings)
    train = 1
    for training_map in trainings:
        print(f"Downloading training {train} of {num_trainings}")
        train += 1
        for layer_map in training_map['layers']:
            if len(layer_map['findings']) > 0:
                image = {}
                image['id'] = layer_map['id']
                image['training_id'] = layer_map['training_id']
                image_with_errors_path = os.path.join(
                    WITH_ERRORS_PATH, f"{layer_map['training_id']}-{layer_map['id']}.png")
                image['with_errors_path'] = image_with_errors_path
                image['without_errors_path'] = os.path.join(
                    WITHOUT_ERRORS_PATH, f"{layer_map['training_id']}-{layer_map['id']}.png")
                findings = []
                for finding_map in layer_map['findings']:
                    if finding_map['status'] != 'discarded':
                        finding = {}
                        finding['id'] = finding_map['id']
                        finding['class'] = finding_map['error_type_id'] -1
                        bounding_box = ((finding_map['x'], finding_map['y']), (
                            finding_map['x'] + finding_map['w'], finding_map['y'] + finding_map['h']))
                        finding['bounding_box'] = bounding_box
                        findings.append(finding)
                        if finding['class'] != None:
                            findings_map[finding['class']].append(
                                image_with_errors_path)

                if len(findings) > 0:
                    download_image(
                        layer_map['original_image_url'], image['with_errors_path'])
                    download_image(
                        layer_map['modified_image_url'], image['without_errors_path'])
                    img = Image.open(image['with_errors_path'])
                    img_size = img.size
                    if img_size[0] > crop_size[0] and img_size[1] > crop_size[1]:
                        image['findings'] = findings
                        image['findings_number'] = len(findings)
                        print(
                            f"Image {layer_map['training_id']}-{layer_map['id']}.png added")
                        images.append(image)
                    else:
                        os.remove(image['without_errors_path'])
                        os.remove(image['with_errors_path'])

    findings_map[0] = list(set(findings_map[0]))
    findings_map[1] = list(set(findings_map[1]))
    findings_map[2] = list(set(findings_map[2]))
    return images, findings_map


def delete_directory(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))


def delete_all_directories():
    delete_directory(DATASET_PATH)


def count_files_in_dir(path):
    return len(os.listdir(path))


def create_directories():
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(WITH_ERRORS_PATH, exist_ok=True)
    os.makedirs(WITHOUT_ERRORS_PATH, exist_ok=True)
    os.makedirs(METADATA_PATH, exist_ok=True)


def create_file(file_name, new_images_maps):
    with open(os.path.join(METADATA_PATH, file_name), 'w') as f:
        for image_map in new_images_maps:
            if len(image_map['findings']) > 0:
                f.write(image_map['without_errors_path'])
                for finding in image_map['findings']:
                    f.write(' ' + ','.join([str(element) for tupl in finding['bounding_box']
                            for element in tupl]) + ',' + str(finding['class']))
                f.write('\n')


def create_finding_file(findings_map):
    for i in range(3):
        with open(os.path.join(METADATA_PATH, f"{i}.txt"), 'w') as f:
            for image_uri in findings_map[i]:
                f.write(image_uri)
                f.write('\n')


def create_data_file(images_map):
    y = json.dumps(images_map)
    with open(os.path.join(METADATA_PATH, f"data.txt"), 'w') as f:
        f.write(y)


def create_dataset():
    trainings = get_trainings_info(SCALE)
    # delete_all_directories()
    create_directories()
    images_map, findings_map = preprocess_data(trainings, CROP_SHAPE)

    images_map_train, images_map_val = train_test_split(images_map, test_size=0.2)
    images_map_val, images_map_test = train_test_split(images_map_val, test_size=0.3)
    create_file("train.txt", images_map_train)

    create_file("val.txt", images_map_val)

    create_file("test.txt", images_map_test)

    create_finding_file(findings_map)
    # create_data_file(images_map)
    # create_data_file(images_map)


if __name__ == '__main__':
    create_dataset()
