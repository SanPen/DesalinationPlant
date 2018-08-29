import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Gui.GuiMain import *

if __name__ == '__main__':
    run()


    # TODO: variar la curva de consumo de agua

    # TODO: Explicar mejor DYCORS, ver artículo

    # TODO: Dar valor a la parte de simulación

    # TODO: Tomar casos prácticos: forzar ceros en los perfiles

    # TODO: Más experimentos

    # TODO: Plotear la función en 3D (función de tamños de solar y eólica, solar y storage, eólica y storage) esto viene de sacar mapa 3D de la función

    # TODO: Comentar mejor el tema realtime