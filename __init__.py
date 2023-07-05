bl_info = {
    "category": "Import-Export",
    "name": "MmdUtil(pmd/pmx)",
    "author": "ousttrue",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "location": "File > Import-Export",
    "description": "Import-Export PMD/PMX meshes",
    "warning": "",
    "wiki_url": "https://github.com/ousttrue/MmdUtility",
    "tracker_url": "https://github.com/ousttrue/MmdUtility",
}


if "bpy" in locals():
    import importlib

    importlib.reload(import_pmx)
    importlib.reload(export_pmx)

import bpy
from . import import_pmx
from . import export_pmx


def register():
    bpy.utils.register_class(import_pmx.ImportPmx)
    bpy.types.TOPBAR_MT_file_import.append(import_pmx.menu_func)
    # bpy.types.INFO_MT_file_export.append(export_pmx.ExportPmx.menu_func)


def unregister():
    bpy.utils.unregister_class(import_pmx.ImportPmx)


if __name__ == "__main__":
    register()

