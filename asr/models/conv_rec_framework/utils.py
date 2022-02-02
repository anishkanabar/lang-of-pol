"""
supplementary methods to facility framework implementation
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

import random


def generate_ramdom_code():
    """
    generate 5 random digit
    """
    code = ''
    for i in range(5):
        code += str(random.randint(0, 9))
    return code




