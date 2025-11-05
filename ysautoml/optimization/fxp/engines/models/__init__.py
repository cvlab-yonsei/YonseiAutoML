#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

## resnet20 ##
from .resnet20 import *

from .mobilenet_ori import *

from .mobilenet_ste import *


def get_model(config):
    print('model name:', config.student_model.name)
    f = globals().get(config.student_model.name)
    if config.student_model.params is None:
        return f()
    else:
        return f(**config.student_model.params)
