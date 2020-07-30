print('Line 1.')

import os
import minkowski_tools as mt

# def touch(path):
#     with open(path, 'a'):
#         os.utime(path, None)

print('After import.')
print(mt.headers)

# test_path = 

with open("Output.txt", "w") as text_file:
    print(mt.headers, file=text_file)

# open(x, test_path).close()