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

    local_map = locals()

    def reload(module_name: str):
        module = local_map.get(module_name)
        if module:
            importlib.reload(module)
        else:
            print(f'{module_name} is not in locals')

    reload("import_pmx")
    reload("export_pmx")

import bpy  # type: ignore
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
