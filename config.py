import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

DICT = os.path.join(DATA_DIR,'dicts/id_full.txt')
STWORDS = os.path.join(DATA_DIR,'dicts/stop_words.txt')



TRAINING_DIR = os.path.join(DATA_DIR,'training_data')
TRAINING_NORMAL_CMT = os.path.join(TRAINING_DIR,'normal_comments.txt')
TRAINING_SARA_CMT = os.path.join(TRAINING_DIR,'sara_comments.txt')

TEST_DATA = os.path.join(DATA_DIR,'test_data')
TEST_NORMAL_CMT = os.path.join(TEST_DATA,'normal_comments.txt')
TEST_SARA_CMT = os.path.join(TEST_DATA,'sata_comments.txt')


