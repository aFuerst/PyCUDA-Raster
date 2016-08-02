# -*- coding: utf-8 -*-
"""
/***************************************************************************
 CUDARaster
                                 A QGIS plugin
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
from PyQt4.QtGui import QAction, QIcon, QFileDialog, QCheckBox, QComboBox
#from qgis.utils import iface
import qgis
from qgis.core import *
from qgis.gui import *
from qgis.utils import *
# Initialize Qt resources from file resources.py
import resources
# Import the code for the dialog
from cudaRaster_dialog import CUDARasterDialog
import os.path

import scheduler

class CUDARaster:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'CUDARaster_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = CUDARasterDialog()

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&CUDA Raster')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'CUDARaster')
        self.toolbar.setObjectName(u'CUDARaster')

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


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToRasterMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/CUDARaster/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'CUDA Raster'),
            callback=self.run,
            parent=self.iface.mainWindow())

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginRasterMenu(
                self.tr(u'&CUDA Raster'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    def select_output_folder(self):
        foldername = QFileDialog.getExistingDirectory(self.dlg, "Select output folder ","")
        self.dlg.output_line.setText(foldername)

    def select_input_file(self):
        filename = QFileDialog.getOpenFileName(self.dlg, "Select input file ","", "Supported files (*.asc *.tif)")
        self.dlg.input_line.setText(filename)

    def run(self):
        from os import name
        """Run method that performs all the real work"""
        # get layers currently loaded in qgis
        self.dlg.input_layer_box.clear()
        self.layers = self.iface.legendInterface().layers()
        layer_list = []
        layer_list.append("None")
        for layer in self.layers:
             layer_list.append(layer.name())
        self.dlg.input_layer_box.addItems(layer_list)
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            functions = []
            outputs = []
            selected_index = self.dlg.input_layer_box.currentIndex()
            print "layer index: ", selected_index

            # Check if layer selected for input
            # If os.name is posix, file structure uses forward slashes. Otherwise,
            # in Windows use back slashes.
            if selected_index != 0:
                input_file = self.layers[selected_index-1]
                input_file_name = input_file.name()
                print "file name: ", input_file_name
                if name == 'posix':
                    input_file_name = "/" + input_file_name
                else:
                    input_file_name = "\\" + input_file_name

            # If no layer already in QGIS was selected, check input from disk.
            else:
                input_file = self.dlg.input_line.text()
                if input_file == "":
                    print "NO OPTION SELECTED!"
                    return
         
                if name == 'posix':
                    input_file_name = input_file[input_file.rfind('/')+1:]
                    input_file_name = "/" + input_file_name[:-4]
                else:
                    input_file_name = input_file[input_file.rfind('\\')+1:]
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
                outputs.append(self.dlg.output_line.text()\
                             + input_file_name\
                             + "_" + function + ".tif") 

            # Run main code
            scheduler.run(input_file, outputs, functions)

            # Load layer back into QGIS if requested
            if self.dlg.slope_check.isChecked() and self.dlg.qgis_slope_check.isChecked():
                for string in outputs:
                    if "_slope" in string:
                        fileInfo = QFileInfo(string)
                        baseName = fileInfo.baseName()            
                        rlayer = QgsRasterLayer(string, baseName)
                        QgsMapLayerRegistry.instance().addMapLayer(rlayer)
            if self.dlg.aspect_check.isChecked() and self.dlg.qgis_aspect_check.isChecked():
                for string in outputs:
                    if "_aspect" in string:
                        fileInfo = QFileInfo(string)
                        baseName = fileInfo.baseName()            
                        rlayer = QgsRasterLayer(string, baseName)
                        QgsMapLayerRegistry.instance().addMapLayer(rlayer)
            if self.dlg.hillshade_check.isChecked() and self.dlg.qgis_hillshade_check.isChecked():
                for string in outputs:
                    if "_hillshade" in string:
                        fileInfo = QFileInfo(string)
                        baseName = fileInfo.baseName()            
                        rlayer = QgsRasterLayer(string, baseName)
                        QgsMapLayerRegistry.instance().addMapLayer(rlayer)
		
