# -*- coding: utf-8 -*-
"""
/***************************************************************************
 LinkerTester
                                 A QGIS plugin
 Test calling an outside script from qgis
                             -------------------
        begin                : 2016-06-16
        copyright            : (C) 2016 by af
        email                : af
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load LinkerTester class from file LinkerTester.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .link_test import LinkerTester
    return LinkerTester(iface)
