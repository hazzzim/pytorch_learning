# target after finishing -> construct a data class using pytorch dataset class
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))