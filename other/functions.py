from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap
import traceback

def choisir_fichier(window):
    # choisir le fichier
    type_fichier = 'image (*)'
    titre = "Choisir une image"        
    file = QFileDialog.getOpenFileName(window, titre, '..', type_fichier)[0]
    # if cancel was pressed, just exit
    if file == '':
        return
    return file

def error_msg(text, title="Erreure", more_info=None):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(text)
    if more_info: msg.setInformativeText(more_info)
    msg.setWindowTitle(title)
    msg.exec_()

def load_image_on_label(label, image_file):
    try:
        label.setPixmap(QPixmap(image_file))
        label.show()
    except:
        error_msg("le chargement de l'image a échoué !", more_info=traceback.format_exc())
        print(traceback.format_exc())