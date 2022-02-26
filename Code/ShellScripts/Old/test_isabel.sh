#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/ImplicitStreamFunction

python -u Code/test.py --cdf true --load_from isabel_4x128
python -u Code/test.py --cdf true --load_from isabel_4x256
python -u Code/test.py --cdf true --load_from isabel_4x512
python -u Code/test.py --cdf true --load_from isabel_4x1024
python -u Code/test.py --cdf true --load_from isabel_4x2048
python -u Code/test.py --cdf true --load_from isabel_6x128
python -u Code/test.py --cdf true --load_from isabel_6x256
python -u Code/test.py --cdf true --load_from isabel_6x512
python -u Code/test.py --cdf true --load_from isabel_6x1024
python -u Code/test.py --cdf true --load_from isabel_6x2048
python -u Code/test.py --cdf true --load_from isabel_8x128
python -u Code/test.py --cdf true --load_from isabel_8x256
python -u Code/test.py --cdf true --load_from isabel_8x512
python -u Code/test.py --cdf true --load_from isabel_8x1024
python -u Code/test.py --cdf true --load_from isabel_8x2048
python -u Code/test.py --cdf true --load_from isabel_10x128
python -u Code/test.py --cdf true --load_from isabel_10x256
python -u Code/test.py --cdf true --load_from isabel_10x512
python -u Code/test.py --cdf true --load_from isabel_10x1024
python -u Code/test.py --cdf true --load_from isabel_10x2048
