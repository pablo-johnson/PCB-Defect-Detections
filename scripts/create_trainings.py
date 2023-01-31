import requests
import json
import os
from pathlib import Path

from zipfile import ZipFile, ZIP_DEFLATED

# Query URL
BASE_URL = 'http://155.138.233.78:8000/api/'
# BASE_URL = 'http://localhost:8000/api/'
LOGIN_ENDPOINT = 'users/login/token/'
CREATE_TRAINING_ENDPOINT = 'trainings/'
UPLOAD_TRAINING_FILES_ENDPOINT = 'trainings/'
USER = 'pjohnson@example.com'
PASSWORD = 'MOlita01!'
TRAININGS_DIR = "/Users/pjohnson/Ilmenau/WS2022/Tesis/data-test/vultr/trainings"
DATA_FILE = "data.json"
NEW_TRAININGS_DIR = "/Users/pjohnson/Downloads/tesis/diseños/nuevos"
NEW_DATA_FILE = "new_data.txt"
PRETRAINING_DIR = "/Users/pjohnson/Downloads/tesis/diseños/pretraining"
PRETRAINING_DATA_FILE = "pretraining_data.txt"

SCALE = 10


def read_json_data(file):
    f = open(file)
    data = json.load(f)
    return data


def read_txt_data(file):
    dirs = []
    with open(file, 'r') as f:
        for line in f:
            line = line.split(";")
            file = {'name': line[0], 'description': f"{line[1]}"}
            dirs.append(file)

    return dirs


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


def create_training(data):
    headers = authenticate()
    url = (f'{BASE_URL}{CREATE_TRAINING_ENDPOINT}')
    res = requests.post(url, headers=headers, json=data)
    training = res.json()
    return training


def upload_files_to_training(training_id, original_file, modified_file):
    # print(original_file.name)
    headers = authenticate()
    # headers['Content-Type'] = 'multipart/form-data'

    url = (f'{BASE_URL}{UPLOAD_TRAINING_FILES_ENDPOINT}{training_id}')
    res = requests.post(url, headers=headers,
                        files={
                            'original_file': ("zip", open(original_file, "rb"),
                                              "multipart/form-data"),
                            "modified_file": ("zip", open(modified_file, "rb"),
                                              "multipart/form-data")})
    print(res)
    training = res.json()
    return training


def create_trainings(trainings):
    num_trainings = 0
    for training in trainings:
        old_id = training["id"]
        created_training = create_training({
            "new_training": {
                "name": training["name"],
                "description": f"{training['description']} ({SCALE}x{SCALE})",
                "scale": SCALE
            }
        })
        path = os.path.join(TRAININGS_DIR, f"{old_id}")
        os.chdir(path)
        fix_zip_files_extension(old_id)
        upload_files_to_training(
            created_training["id"],
            f"{old_id}-original.zip",
            f"{old_id}-modified.zip")
        num_trainings += 1
        print(f"training {created_training['id']} done.\n")

    print(f"{num_trainings} trainings created.")

def fix_zip_files_extension(id):
    p = Path(f'{id}-original.original_file')
    if p.exists():
        p.rename(p.with_suffix('.zip'))
    p = Path(f'{id}-modified.original_file')
    if p.exists():
        p.rename(p.with_suffix('.zip'))


def unzip_and_zip_files(training, zip_file, output_zip_file):
    path = os.path.join(NEW_TRAININGS_DIR, f"{training['name']}")
    os.chdir(path)
    with ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("./temp")

    zipObj = ZipFile(output_zip_file, 'w', compression=ZIP_DEFLATED)
    os.chdir(os.path.join(path, "temp"))
    path = os.path.join(NEW_TRAININGS_DIR, f"{training['name']}", zip_file)
    zipObj.write('Gerber_BottomLayer.GBL')
    zipObj.write('Gerber_TopLayer.GTL')
    zipObj.close()


def create_new_trainings(new_trainings):
    num_trainings = 0
    for training in new_trainings:
        print(f"Creating now {training['name']}")
        created_training = create_training({
            "new_training": {
                "name": training["name"],
                "description": f"{training['description']} ({SCALE}x{SCALE})",
                "scale":SCALE
            }
        })

        unzip_and_zip_files(training, "with_errors.zip", "original.zip")
        unzip_and_zip_files(training, "without_errors.zip", "modified.zip")
        path = os.path.join(NEW_TRAININGS_DIR, f"{training['name']}")
        os.chdir(path)
        upload_files_to_training(
            created_training["id"],
            f"original.zip",
            f"modified.zip")
        num_trainings += 1
        print(f"training {created_training['id']} done.\n")

    print(f"{num_trainings} trainings created.")


def unzip_and_zip_files2(zip_filename, output_zip_filename):
    with ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall("./temp")

    zipObj = ZipFile(output_zip_filename, 'w', compression=ZIP_DEFLATED)
    os.chdir(os.path.join("./temp"))
    zipObj.write('Gerber_BottomLayer.GBL')
    zipObj.write('Gerber_TopLayer.GTL')
    zipObj.close()


def create_pretrainings(pretrainings):
    num_trainings = 0
    for training in pretrainings:
        created_training = create_training({
            "new_training": {
                "name": training["name"],
                "description": training["description"]
            }
        })
        os.chdir(PRETRAINING_DIR)
        unzip_and_zip_files2(f"{training['name']}.zip",
                             f"{created_training['id']}.zip")
        os.chdir(PRETRAINING_DIR)
        upload_files_to_training(
            created_training["id"],
            f"{created_training['id']}.zip",
            f"{created_training['id']}.zip")

        num_trainings += 1
        print(f"training {created_training['id']} done.\n")

    print(f"{num_trainings} trainings created.")


def start_flow():
    trainings = read_json_data(DATA_FILE)
    create_trainings(trainings)
    # os.chdir(os.path.join("/Users/pjohnson/Ilmenau/WS2022/Tesis/scripts"))
    # new_trainings = read_txt_data(NEW_DATA_FILE)
    # create_new_trainings(new_trainings)
    # os.chdir(os.path.join("/Users/pjohnson/Ilmenau/WS2022/Tesis/scripts"))
    # pretrainings = read_txt_data(PRETRAINING_DATA_FILE)
    # create_pretrainings(pretrainings)


if __name__ == '__main__':
    start_flow()
