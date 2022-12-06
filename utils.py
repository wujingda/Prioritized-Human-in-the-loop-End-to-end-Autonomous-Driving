import sys
import pygame
def signal_handler(sig, frame):
    print('Procedure terminated!')
    pygame.display.quit()
    pygame.quit()
    sys.exit(0)

import torch
import numpy as np
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
