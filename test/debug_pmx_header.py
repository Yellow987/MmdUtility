#!/usr/bin/env python
# coding: utf-8
"""
PMX Header Diagnostic Tool
==========================

This script analyzes PMX file headers to identify format differences
and understand why the Idolmaster PMX file is failing to load.
"""

import io
import os
import sys
import struct

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from pymeshio import common


def analyze_pmx_header(file_path):
    """Analyze PMX file header structure."""
    print(f"Analyzing PMX file: {file_path}")
    print("=" * 50)
    
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
            
        print(f"File size: {len(data)} bytes")
        
        # Create binary reader
        reader = common.BinaryReader(io.BytesIO(data))
        
        # Read signature
        try:
            signature = reader.unpack("4s", 4)
            print(f"Signature: {signature} (should be b'PMX ')")
            
            if signature != b"PMX ":
                print(f"WARNING: Invalid signature! Expected b'PMX ', got {signature}")
                # Try to find PMX signature in the file
                pmx_pos = data.find(b"PMX ")
                if pmx_pos != -1:
                    print(f"Found 'PMX ' at position {pmx_pos}")
                    # Skip to PMX signature and try again
                    reader = common.BinaryReader(io.BytesIO(data[pmx_pos:]))
                    signature = reader.unpack("4s", 4)
                    print(f"New signature: {signature}")
                else:
                    print("No PMX signature found in file!")
                    return
        except Exception as e:
            print(f"Error reading signature: {e}")
            return
            
        # Read version
        try:
            version = reader.read_float()
            print(f"Version: {version}")
        except Exception as e:
            print(f"Error reading version: {e}")
            return
            
        # Read flag bytes count
        try:
            flag_bytes = reader.read_int(1)
            print(f"Flag bytes count: {flag_bytes} (should be 8)")
            
            if flag_bytes != 8:
                print(f"WARNING: Unexpected flag bytes count! Expected 8, got {flag_bytes}")
                # This might be the issue - let's see what happens if we continue
                
        except Exception as e:
            print(f"Error reading flag bytes count: {e}")
            return
            
        # Try to read the flags even if count is wrong
        try:
            print("\nReading format flags:")
            text_encoding = reader.read_int(1)
            print(f"  Text encoding: {text_encoding}")
            
            extended_uv = reader.read_int(1) 
            print(f"  Extended UV: {extended_uv}")
            
            vertex_index_size = reader.read_int(1)
            print(f"  Vertex index size: {vertex_index_size}")
            
            texture_index_size = reader.read_int(1)
            print(f"  Texture index size: {texture_index_size}")
            
            material_index_size = reader.read_int(1)
            print(f"  Material index size: {material_index_size}")
            
            bone_index_size = reader.read_int(1)
            print(f"  Bone index size: {bone_index_size}")
            
            morph_index_size = reader.read_int(1)
            print(f"  Morph index size: {morph_index_size}")
            
            rigidbody_index_size = reader.read_int(1)
            print(f"  Rigidbody index size: {rigidbody_index_size}")
            
            # If we have more than 8 flag bytes, read the extras
            extra_flags = []
            remaining_flags = max(0, flag_bytes - 8)
            for i in range(remaining_flags):
                extra_flag = reader.read_int(1)
                extra_flags.append(extra_flag)
                print(f"  Extra flag {i+1}: {extra_flag}")
                
        except Exception as e:
            print(f"Error reading format flags: {e}")
            return
            
        print(f"\nCurrent reader position: {reader.ios.tell()}")
        
        # Try to read model info
        try:
            print("\nReading model info:")
            
            # Create a test reader with the flags we read
            from pymeshio.pmx.reader import Reader
            
            test_reader = Reader(reader.ios,
                               text_encoding,
                               extended_uv, 
                               vertex_index_size,
                               texture_index_size,
                               material_index_size,
                               bone_index_size,
                               morph_index_size,
                               rigidbody_index_size)
            
            # Try reading model name
            model_name = test_reader.read_text()
            print(f"  Model name: {model_name}")
            
            english_name = test_reader.read_text()
            print(f"  English name: {english_name}")
            
            comment = test_reader.read_text()
            print(f"  Comment: {comment[:100]}...")
            
            english_comment = test_reader.read_text()
            print(f"  English comment: {english_comment[:100]}...")
            
            # Try reading vertex count
            vertex_count = test_reader.read_int(4)
            print(f"  Vertex count: {vertex_count}")
            
            if vertex_count > 1000000:  # Suspiciously large
                print(f"WARNING: Vertex count seems too large: {vertex_count}")
                print("This might be where the parsing error occurs!")
                
            # Try to read the first vertex to see if that's where it fails
            try:
                print(f"Attempting to read vertices (count: {vertex_count})...")
                vertex = test_reader.read_vertex()
                print(f"Successfully read first vertex: position=({vertex.position.x:.3f}, {vertex.position.y:.3f}, {vertex.position.z:.3f})")
                
                # Try reading index count
                indices_count = test_reader.read_int(4)
                print(f"Indices count: {indices_count}")
                
                if indices_count > 100000000:  # Suspiciously large
                    print(f"WARNING: Indices count seems too large: {indices_count}")
                    print("This is likely where the 1GB allocation error occurs!")
                    
                # Try reading texture count
                texture_count = test_reader.read_int(4)
                print(f"Texture count: {texture_count}")
                
                # Try reading material count
                material_count = test_reader.read_int(4)
                print(f"Material count: {material_count}")
                
            except Exception as ve:
                print(f"Error reading vertex/index data: {ve}")
                print("This is where the original parsing error likely occurs!")
                
        except Exception as e:
            print(f"Error reading model info: {e}")
            print("This is likely where the original error occurs")
            
    except Exception as e:
        print(f"Fatal error analyzing file: {e}")


def hex_dump(data, start=0, length=64):
    """Print hex dump of data."""
    print(f"Hex dump (offset {start}, {length} bytes):")
    for i in range(0, min(length, len(data) - start), 16):
        offset = start + i
        hex_part = ' '.join(f'{data[offset + j]:02x}' for j in range(min(16, len(data) - offset)))
        ascii_part = ''.join(chr(data[offset + j]) if 32 <= data[offset + j] <= 126 else '.' 
                           for j in range(min(16, len(data) - offset)))
        print(f"{offset:08x}: {hex_part:<48} {ascii_part}")


def main():
    """Main diagnostic function."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    pmx_file = os.path.join(test_dir, 'pdtt.pmx')
    
    if not os.path.exists(pmx_file):
        print(f"PMX file not found: {pmx_file}")
        return
        
    # Show first 128 bytes as hex dump
    with open(pmx_file, 'rb') as f:
        data = f.read(128)
        
    hex_dump(data)
    print()
    
    # Analyze the header
    analyze_pmx_header(pmx_file)


if __name__ == "__main__":
    main()