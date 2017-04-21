# coding: utf-8
import os
import sys
import time
import functools

try:
    import bpy
    import mathutils
except:
    pass


FS_ENCODING=sys.getfilesystemencoding()
if os.path.exists(os.path.dirname(sys.argv[0])+"/utf8"):
    INTERNAL_ENCODING='utf-8'
else:
    INTERNAL_ENCODING=FS_ENCODING


def createVector(x, y, z):
    return mathutils.Vector([x, y, z])


class object:
    @staticmethod
    def createEmpty(scene, name):
        empty=bpy.data.objects.new(name, None)
        scene.objects.link(empty)
        return empty

    @staticmethod
    def duplicate(scene, o):
        bpy.ops.object.select_all(action='DESELECT')
        o.select=True
        scene.objects.active=o
        bpy.ops.object.duplicate()
        dumy=scene.objects.active
        #bpy.ops.object.rotation_apply()
        #bpy.ops.object.scale_apply()
        #bpy.ops.object.location_apply()
        dumy.data.update(calc_tessface=True)
        return dumy.data, dumy

    @staticmethod
    def setLayerMask(object, layers):
        layer=[]
        for i in range(20):
            try:
                layer.append(True if layers[i]!=0 else False)
            except IndexError:
                layer.append(False)
        object.layers=layer

    @staticmethod
    def getShapeKeys(o):
        return o.data.shape_keys.key_blocks

    @staticmethod
    def addShapeKey(o, name):
        try:
            return o.shape_key_add(name)
        except:
            return o.add_shape_key(name)

    @staticmethod
    def getVertexGroup(o, name):
        indices=[]
        for i, v in enumerate(o.data.vertices):
            for g in v.groups:
                if o.vertex_groups[g.group].name==name:
                    indices.append(i)
        return indices

    @staticmethod
    def getVertexGroupNames(o):
        for g in o.vertex_groups:
            yield g.name

    @staticmethod
    def addVertexGroup(o, name):
        o.vertex_groups.new(name)

    @staticmethod
    def assignVertexGroup(o, name, index, weight):
        if name not in o.vertex_groups:
            o.vertex_groups.new(name)
        o.vertex_groups[name].add([index], weight, 'ADD')

    @staticmethod
    def createBoneGroup(scene, o, name, color_set='DEFAULT'):
        # create group
        o.select=True 
        scene.objects.active=o
        bpy.ops.object.mode_set(mode='POSE', toggle=False)
        bpy.ops.pose.group_add()
        # set name
        pose=o.pose
        g=pose.bone_groups.active
        g.name=name
        g.color_set=color_set
        return g


"""
custom property keys
"""
MMD_SHAPE_GROUP_NAME='_MMD_SHAPE'

MMD_MB_NAME='mb_name'
MMD_ENGLISH_NAME='english_name'

MMD_MB_COMMENT='mb_comment'
MMD_COMMENT='comment'
MMD_ENGLISH_COMMENT='english_comment'

BONE_ENGLISH_NAME='english_name'
BONE_USE_TAILOFFSET='bone_use_tailoffset'
BONE_CAN_TRANSLATE='bone_can_translate'
IK_UNITRADIAN='ik_unit_radian'

BASE_SHAPE_NAME='Basis'
RIGID_NAME='rigid_name'
RIGID_SHAPE_TYPE='rigid_shape_type'
RIGID_PROCESS_TYPE='rigid_process_type'
RIGID_BONE_NAME='rigid_bone_name'
RIGID_GROUP='ribid_group'
RIGID_INTERSECTION_GROUP='rigid_intersection_group'
RIGID_WEIGHT='rigid_weight'
RIGID_LINEAR_DAMPING='rigid_linear_damping'
RIGID_ANGULAR_DAMPING='rigid_angular_damping'
RIGID_RESTITUTION='rigid_restitution'
RIGID_FRICTION='rigid_friction'
CONSTRAINT_NAME='const_name'
CONSTRAINT_A='const_a'
CONSTRAINT_B='const_b'
CONSTRAINT_POS_MIN='const_pos_min'
CONSTRAINT_POS_MAX='const_pos_max'
CONSTRAINT_ROT_MIN='const_rot_min'
CONSTRAINT_ROT_MAX='const_rot_max'
CONSTRAINT_SPRING_POS='const_spring_pos'
CONSTRAINT_SPRING_ROT='const_spring_rot'
TOON_TEXTURE_OBJECT='ToonTextures'

MATERIALFLAG_BOTHFACE='material_flag_bothface'
MATERIALFLAG_GROUNDSHADOW='material_flag_groundshadow'
MATERIALFLAG_SELFSHADOWMAP='material_flag_selfshadowmap'
MATERIALFLAG_SELFSHADOW='material_flag_drawselfshadow'
MATERIALFLAG_EDGE='material_flag_drawedge'
MATERIAL_SHAREDTOON='material_shared_toon'
MATERIAL_SPHERE_MODE='material_sphere_mode'
TEXTURE_TYPE='texture_type'

