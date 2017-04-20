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


class modifier:
    @staticmethod
    def addMirror(mesh_object):
        return mesh_object.modifiers.new("Modifier", "MIRROR")

    @staticmethod
    def addArmature(mesh_object, armature_object):
        mod=mesh_object.modifiers.new("Modifier", "ARMATURE")
        mod.object = armature_object
        mod.use_bone_envelopes=False

    @staticmethod
    def hasType(mesh_object, type_name):
        for mod in mesh_object.modifiers:
                if mod.type==type_name.upper():
                    return True

    @staticmethod
    def isType(m, type_name):
        return m.type==type_name.upper()

    @staticmethod
    def getArmatureObject(m):
        return m.object


class shapekey:
    @staticmethod
    def assign(shapeKey, index, pos):
        shapeKey.data[index].co=pos

    @staticmethod
    def getByIndex(b, index):
        return b.data[index].co

    @staticmethod
    def get(b):
        for k in b.data:
            yield k.co


class texture:
    @staticmethod
    def create(path):
        texture=bpy.data.textures.new(os.path.basename(path), 'IMAGE')
        texture.use_mipmap=True
        texture.use_interpolation=True
        texture.use_alpha=True
        try:
            image=bpy.data.images.load(path)
        except RuntimeError:
            print('fail to create:', path)
            image=bpy.data.images.new('Image', width=16, height=16)
        texture.image=image
        return texture, image

    @staticmethod
    def getPath(t):
        if  t.type=="IMAGE":
            image=t.image
            if image:
                return image.filepath


class material:
    @staticmethod
    def create(name):
        return bpy.data.materials.new(name)

    @staticmethod
    def get(material_name):
        return bpy.data.materials[material_name]

    @staticmethod
    def addTexture(material, texture, enable=True, blend_type='MULTIPLY'):
        # search free slot
        index=None
        for i, slot in enumerate(material.texture_slots):
            if not slot:
                index=i
                break
        if index==None:
            return
        material.use_shadeless=True
        #
        slot=material.texture_slots.create(index)
        slot.texture=texture
        slot.texture_coords='UV'
        slot.blend_type=blend_type
        slot.use_map_alpha=True
        slot.use=enable
        return index

    @staticmethod
    def getTexture(m, index):
        return m.texture_slots[index].texture

    @staticmethod
    def hasTexture(m):
        return m.texture_slots[0]

    @staticmethod
    def setUseTexture(m, index, enable):
        m.use_textures[index]=enable

    @staticmethod
    def eachTexturePath(m):
        for slot in m.texture_slots:
            if slot and slot.texture:
                texture=slot.texture
                if  texture.type=="IMAGE":
                    image=texture.image
                    if not image:
                        continue
                    yield image.filepath

    @staticmethod
    def eachEnalbeTexturePath(m):
        for i, slot in enumerate(m.texture_slots):
            if m.use_textures[i] and slot and slot.texture:
                texture=slot.texture
                if  texture.type=="IMAGE":
                    image=texture.image
                    if not image:
                        continue
                    yield image.filepath

    @staticmethod
    def eachEnalbeTexture(m):
        for i, slot in enumerate(m.texture_slots):
            if m.use_textures[i] and slot and slot.texture:
                texture=slot.texture
                if  texture.type=="IMAGE":
                    image=texture.image
                    if not image:
                        continue
                    yield slot.texture


class mesh:
    @staticmethod
    def create(scene, name):
        mesh=bpy.data.meshes.new("Mesh")
        mesh_object= bpy.data.objects.new(name, mesh)
        scene.objects.link(mesh_object)
        return mesh, mesh_object

    @classmethod
    def addGeometry(cls, mesh, vertices, faces):
        from bpy_extras.io_utils import unpack_list, unpack_face_list
        mesh.vertices.add(len(vertices))
        mesh.vertices.foreach_set("co", unpack_list(vertices))
        mesh.tessfaces.add(len(faces))
        mesh.tessfaces.foreach_set("vertices_raw", unpack_face_list(faces))
        #mesh.from_pydata(vertices, [], faces)
        """
        mesh.add_geometry(len(vertices), 0, len(faces))
        # add vertex
        unpackedVertices=[]
        for v in vertices:
            unpackedVertices.extend(v)
        mesh.vertices.foreach_set("co", unpackedVertices)
        # add face
        unpackedFaces = []
        for face in faces:
            if len(face) == 4:
                if face[3] == 0:
                    # rotate indices if the 4th is 0
                    face = [face[3], face[0], face[1], face[2]]
            elif len(face) == 3:
                if face[2] == 0:
                    # rotate indices if the 3rd is 0
                    face = [face[2], face[0], face[1], 0]
                else:
                    face.append(0)
            unpackedFaces.extend(face)
        mesh.faces.foreach_set("verts_raw", unpackedFaces)
        """
        assert(len(vertices)==len(mesh.vertices))
        #assert(len(faces)==len(cls.getFaces(mesh)))

    @staticmethod
    def hasUV(mesh):
        return len(mesh.tessface_uv_textures)>0

    @staticmethod
    def useVertexUV(mesh):
        pass

    @staticmethod
    def addUV(mesh):
        mesh.tessface_uv_textures.new()

    @staticmethod
    def hasFaceUV(mesh, i, face):
        active_uv_texture=None
        for t in mesh.tessface_uv_textures:
            if t.active:
                active_uv_texture=t
                break
        return active_uv_texture and active_uv_texture.data[i]

    @staticmethod
    def getFaceUV(mesh, i, faces, count=3):
        active_uv_texture=None
        for t in mesh.tessface_uv_textures:
            if t.active:
                active_uv_texture=t
                break
        if active_uv_texture and active_uv_texture.data[i]:
            uvFace=active_uv_texture.data[i]
            if count==3:
                return (uvFace.uv1, uvFace.uv2, uvFace.uv3)
            elif count==4:
                return (uvFace.uv1, uvFace.uv2, uvFace.uv3, uvFace.uv4)
            else:
                print(count)
                assert(False)
        else:
            return ((0, 0), (0, 0), (0, 0), (0, 0))


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

