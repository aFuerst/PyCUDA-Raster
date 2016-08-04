# -*- coding: utf-8 -*-
"""
/***************************************************************************
 CUDARaster
 Utilize NVIDIA GPU to do raster calculations
                              -------------------
        begin                : 2016-07-15
        git sha              : $Format:%H$
        copyright            : (C) 2016 by Alex Feurst, Charles Kazer, William Hoffman
        email                : ckazer1@swarthmore.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from PyQt4.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, QFileInfo
from PyQt4.QtGui import QAction, QIcon, QFileDialog, QCheckBox, QComboBox, QApplication
# Import the code for the dialog
from cudaRaster_dialog import CUDARasterDialog
from cudaRasterCrash_dialog import CUDARasterDialogCrash
import os.path

import scheduler

class CUDARaster:

    def __init__(self):#, iface):
        # initialize plugin directory
        self._dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')#[0:2]
        locale_path = os.path.join(
            self._dir,
            'i18n',
            'CUDARaster_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = CUDARasterDialog()
        self.dlg2 = CUDARasterDialogCrash()

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&CUDA Raster')

        self.dlg.input_line.clear()
        self.dlg.input_button.clicked.connect(self.select_input_file)
        
        self.dlg.output_line.clear()
        self.dlg.output_button.clicked.connect(self.select_output_folder)

        self.dlg.slope_check.setChecked(False)
        self.dlg.aspect_check.setChecked(False)
        self.dlg.hillshade_check.setChecked(False)


    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('CUDARaster', message)

    def select_output_folder(self):
        foldername = QFileDialog.getExistingDirectory(self.dlg, "Select output folder ","")
        self.dlg.output_line.setText(foldername)

    def select_input_file(self):
        filename = QFileDialog.getOpenFileName(self.dlg, "Select input file ","", "Supported files (*.asc *.tif)")
        self.dlg.input_line.setText(filename)

    def run(self):
        from os import name
        """Run method that performs all the real work"""
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            functions = []
            outputs = []

            # If os.name is posix, file structure uses forward slashes. Otherwise,
            # in Windows use back slashes.
            input_file = str(self.dlg.input_line.text())
            if input_file == "":
                print "NO OPTION SELECTED!"
                return

            if name == 'posix':
                input_file_name = input_file[input_file.rfind('/')+1:]
                input_file_name = "/" + input_file_name[:-4]
            else:
                input_file_name = input_file[input_file.rfind('/')+1:]
                input_file_name = "\\" + input_file_name[:-4]

            print "file name: ", input_file_name
            print "input file: ", input_file

            print input_file, " in cudaRaster"
            if self.dlg.slope_check.isChecked():
                functions.append("slope")
            if self.dlg.aspect_check.isChecked():
                functions.append("aspect")
            if self.dlg.hillshade_check.isChecked():
                functions.append("hillshade")
            for function in functions:
                outputs.append(str(self.dlg.output_line.text())\
                             + input_file_name\
                             + "_" + function + ".tif") 

            # Run main code
            if scheduler.run(input_file, outputs, functions):
                print "Something went wrong."
                self.dlg2.show()
                self.dlg2.pushButton.clicked.connect((lambda: self.dlg2.close()))
		
if __name__=="__main__":
	import sys
	
	app = QApplication(sys.argv)
	dialog = CUDARaster()
	dialog.run()
