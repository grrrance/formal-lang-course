import os


def get_path_to_test_case(file_name):
    directory = os.path.join("tests", "test_cases")
    return os.path.join(directory, file_name)
