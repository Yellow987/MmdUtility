#!/usr/bin/env python
# coding: utf-8
"""
PMX Section-by-Section Parser
=============================

This script parses PMX file section by section to identify exactly where
the parsing error occurs and what's causing the 3-byte shortfall.
"""

import io
import os
import sys
import struct

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from pymeshio import common
from pymeshio.pmx.reader import Reader
import pymeshio.pmx as pmx


def debug_pmx_sections(file_path):
    """Parse PMX file section by section with detailed logging."""
    print(f"Parsing PMX file: {file_path}")
    print("=" * 60)
    
    with open(file_path, 'rb') as f:
        data = f.read()
    
    print(f"File size: {len(data)} bytes")
    reader = common.BinaryReader(io.BytesIO(data))
    
    try:
        # Header
        signature = reader.unpack("4s", 4)
        print(f"Signature: {signature}")
        
        version = reader.read_float()
        print(f"Version: {version}")
        
        flag_bytes = reader.read_int(1)
        print(f"Flag bytes: {flag_bytes}")
        
        # Format flags
        text_encoding = reader.read_int(1)
        extended_uv = reader.read_int(1)
        vertex_index_size = reader.read_int(1)
        texture_index_size = reader.read_int(1)
        material_index_size = reader.read_int(1)
        bone_index_size = reader.read_int(1)
        morph_index_size = reader.read_int(1)
        rigidbody_index_size = reader.read_int(1)
        
        print(f"Extended UV: {extended_uv}")
        print(f"Current position: {reader.ios.tell()}")
        
        # Create PMX reader
        pmx_reader = Reader(reader.ios, text_encoding, extended_uv, 
                          vertex_index_size, texture_index_size, 
                          material_index_size, bone_index_size, 
                          morph_index_size, rigidbody_index_size)
        
        # Model info
        print("\n--- MODEL INFO ---")
        model_name = pmx_reader.read_text()
        print(f"Model name: {model_name}")
        print(f"Position after name: {reader.ios.tell()}")
        
        english_name = pmx_reader.read_text()
        print(f"English name: {english_name}")
        print(f"Position after english name: {reader.ios.tell()}")
        
        comment = pmx_reader.read_text()
        print(f"Comment length: {len(comment)}")
        print(f"Position after comment: {reader.ios.tell()}")
        
        english_comment = pmx_reader.read_text()
        print(f"English comment length: {len(english_comment)}")
        print(f"Position after english comment: {reader.ios.tell()}")
        
        # Vertices
        print("\n--- VERTICES ---")
        vertex_count = pmx_reader.read_int(4)
        print(f"Vertex count: {vertex_count}")
        print(f"Position after vertex count: {reader.ios.tell()}")
        
        # Read a few vertices to check alignment
        for i in range(min(3, vertex_count)):
            vertex = pmx_reader.read_vertex()
            print(f"Vertex {i}: pos=({vertex.position.x:.3f}, {vertex.position.y:.3f}, {vertex.position.z:.3f})")
            print(f"Position after vertex {i}: {reader.ios.tell()}")
        
        # Skip remaining vertices
        for i in range(3, vertex_count):
            pmx_reader.read_vertex()
        
        print(f"Position after all vertices: {reader.ios.tell()}")
        
        # Indices
        print("\n--- INDICES ---")
        indices_count = pmx_reader.read_int(4)
        print(f"Indices count: {indices_count}")
        print(f"Position after indices count: {reader.ios.tell()}")
        
        # Skip indices
        for i in range(indices_count):
            pmx_reader.read_vertex_index()
        print(f"Position after all indices: {reader.ios.tell()}")
        
        # Textures
        print("\n--- TEXTURES ---")
        texture_count = pmx_reader.read_int(4)
        print(f"Texture count: {texture_count}")
        print(f"Position after texture count: {reader.ios.tell()}")
        
        for i in range(texture_count):
            texture = pmx_reader.read_text()
            print(f"Texture {i}: {texture}")
        print(f"Position after all textures: {reader.ios.tell()}")
        
        # Materials
        print("\n--- MATERIALS ---")
        material_count = pmx_reader.read_int(4)
        print(f"Material count: {material_count}")
        print(f"Position after material count: {reader.ios.tell()}")
        
        for i in range(material_count):
            material = pmx_reader.read_material()
            print(f"Material {i}: {material.name}")
            print(f"Position after material {i}: {reader.ios.tell()}")
        
        # Bones
        print("\n--- BONES ---")
        bone_count = pmx_reader.read_int(4)
        print(f"Bone count: {bone_count}")
        print(f"Position after bone count: {reader.ios.tell()}")
        
        for i in range(bone_count):
            bone = pmx_reader.read_bone()
            print(f"Bone {i}: {bone.name}")
            print(f"Position after bone {i}: {reader.ios.tell()}")
            if i >= 5:  # Show first few bones only
                break
                
        # Skip remaining bones
        for i in range(6, bone_count):
            pmx_reader.read_bone()
        
        print(f"Position after all bones: {reader.ios.tell()}")
        print(f"Bytes remaining: {len(data) - reader.ios.tell()}")
        
        # Try remaining sections
        print("\n--- REMAINING SECTIONS ---")
        try:
            morph_count = pmx_reader.read_int(4)
            print(f"Morph count: {morph_count}")
            print(f"Position after morph count: {reader.ios.tell()}")
            print(f"Bytes remaining: {len(data) - reader.ios.tell()}")
        except Exception as e:
            print(f"Error reading morph count: {e}")
            print(f"Position when error occurred: {reader.ios.tell()}")
            print(f"Bytes remaining: {len(data) - reader.ios.tell()}")
            
    except Exception as e:
        print(f"Error: {e}")
        print(f"Position when error occurred: {reader.ios.tell()}")
        print(f"Bytes remaining: {len(data) - reader.ios.tell()}")


def main():
    """Main function."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    pmx_file = os.path.join(test_dir, 'pdtt.pmx')
    
    if not os.path.exists(pmx_file):
        print(f"PMX file not found: {pmx_file}")
        return
        
    debug_pmx_sections(pmx_file)


if __name__ == "__main__":
    main()