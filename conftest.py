import pytest


def pytest_addoption(parser):
    parser.addoption("--kaggle2015_folder", action="store")


@pytest.fixture
def kaggle2015_folder(request):
    kaggle2015_folder_value = request.config.getoption("kaggle2015_folder")
    if kaggle2015_folder_value is None:
        pytest.skip("Skipping. Kaggle 2015 folder not set. Use --kaggle2015_folder to set it.")
    return kaggle2015_folder_value
