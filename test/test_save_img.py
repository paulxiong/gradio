# add increamtal save images
from glob import glob
import os
import unittest
import unittest.mock as mock
import urllib
import warnings
class TestFiles(unittest.TestCase):
    suffixx, sep, dirr=".jpg", "-", "run_FineTune_tmp_imgs/im"
    breakpoint()
    files = glob(dirr+"*"+suffixx)
    files = [f.split(".jpg")[0] for f in files]
    files = [int(f.split("-")[-1]) for f in files]
    if files == []:
        cur_num = 0
    else:
        files.sort()
        cur_num=files[-1]
        cur_num += 1
    save_full_path = f"run_FineTune_tmp_imgs/im-{cur_num}.png"
    #m1.save(fsave_full_path)
    warnings.warn(save_full_path)   
if __name__ == "__main__":
    unittest.main()
