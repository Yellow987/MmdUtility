#!/usr/bin/env python3
"""
Test script to properly decode VMD bone names and match with core bones.
"""

import os
import sys

def analyze_vmd_bone_names():
    """Analyze VMD bone names with proper UTF decoding."""
    print("Analyzing VMD Bone Names with Proper UTF Decoding...")
    
    vmd_path = "test/dan_alivef_01.imo.vmd"
    
    if not os.path.exists(vmd_path):
        print(f"✗ VMD test file not found: {vmd_path}")
        return False
    
    try:
        import bone_position_extractor
        extractor = bone_position_extractor.BonePositionExtractor()
        
        print("Loading VMD motion...")
        extractor.load_vmd(vmd_path)
        
        # Define core bones (Japanese names)
        core_bone_names = {
            "", "センター", "グルーブ", "腰", "上半身", "上半身2",
            "首", "頭",
            "左肩", "右肩", "左腕", "右腕", "左ひじ", "右ひじ", "左手首", "右手首",
            "左足", "右足", "左ひざ", "右ひざ", "左足首", "右足首", "左つま先", "右つま先",
        }
        
        # Collect all unique bone names with different decodings
        bone_names_raw = set()
        bone_names_utf8 = set()
        bone_names_cp932 = set()
        bone_names_shiftjis = set()
        
        for motion in extractor.vmd_motion.motions:
            bone_name_raw = motion.name
            bone_names_raw.add(bone_name_raw)
            
            # Try different decodings
            if isinstance(bone_name_raw, bytes):
                try:
                    bone_names_utf8.add(bone_name_raw.decode('utf-8', errors='ignore'))
                except:
                    pass
                
                try:
                    bone_names_cp932.add(bone_name_raw.decode('cp932', errors='ignore'))
                except:
                    pass
                    
                try:
                    bone_names_shiftjis.add(bone_name_raw.decode('shift_jis', errors='ignore'))
                except:
                    pass
            else:
                # Already a string
                bone_names_utf8.add(bone_name_raw)
                bone_names_cp932.add(bone_name_raw)
                bone_names_shiftjis.add(bone_name_raw)
        
        print(f"\nFound {len(bone_names_raw)} unique raw bone names")
        
        print(f"\n" + "="*60)
        print("BONE NAME DECODING COMPARISON")
        print("="*60)
        
        # Show first 10 raw names with all decodings
        raw_names_list = list(bone_names_raw)[:10]
        
        print(f"{'Raw Bytes':<20} | {'UTF-8':<15} | {'CP932':<15} | {'Shift_JIS':<15}")
        print("-" * 70)
        
        for raw_name in raw_names_list:
            if isinstance(raw_name, bytes):
                try:
                    utf8_name = raw_name.decode('utf-8', errors='replace')
                except:
                    utf8_name = "DECODE_ERROR"
                
                try:
                    cp932_name = raw_name.decode('cp932', errors='replace')
                except:
                    cp932_name = "DECODE_ERROR"
                
                try:
                    shiftjis_name = raw_name.decode('shift_jis', errors='replace')
                except:
                    shiftjis_name = "DECODE_ERROR"
                    
                raw_display = str(raw_name)[:18]
                print(f"{raw_display:<20} | {utf8_name:<15} | {cp932_name:<15} | {shiftjis_name:<15}")
            else:
                # Already a string
                print(f"{str(raw_name)[:18]:<20} | {raw_name:<15} | {raw_name:<15} | {raw_name:<15}")
        
        # Check which decoding gives us the most matches with core bones
        print(f"\n" + "="*60)
        print("CORE BONE MATCHING")
        print("="*60)
        
        utf8_matches = core_bone_names & bone_names_utf8
        cp932_matches = core_bone_names & bone_names_cp932
        shiftjis_matches = core_bone_names & bone_names_shiftjis
        
        print(f"UTF-8 matches: {len(utf8_matches)}")
        print(f"CP932 matches: {len(cp932_matches)}")
        print(f"Shift_JIS matches: {len(shiftjis_matches)}")
        
        # Show the matches
        best_matches = cp932_matches if len(cp932_matches) >= len(utf8_matches) else utf8_matches
        best_encoding = "CP932" if len(cp932_matches) >= len(utf8_matches) else "UTF-8"
        
        if len(shiftjis_matches) > len(best_matches):
            best_matches = shiftjis_matches
            best_encoding = "Shift_JIS"
        
        print(f"\nBest encoding: {best_encoding} with {len(best_matches)} matches")
        print("Matching core bones found:")
        for bone_name in sorted(best_matches):
            display_name = bone_name if bone_name else "(root)"
            print(f"  {display_name}")
        
        # Show missing core bones
        missing_bones = core_bone_names - best_matches
        if missing_bones:
            print(f"\nMissing core bones ({len(missing_bones)}):")
            for bone_name in sorted(missing_bones):
                display_name = bone_name if bone_name else "(root)"
                print(f"  {display_name}")
        
        # Show all decoded bone names for the best encoding
        print(f"\n" + "="*60)
        print(f"ALL BONE NAMES ({best_encoding} DECODING)")
        print("="*60)
        
        if best_encoding == "CP932":
            all_names = sorted(bone_names_cp932)
        elif best_encoding == "Shift_JIS":
            all_names = sorted(bone_names_shiftjis)
        else:
            all_names = sorted(bone_names_utf8)
        
        for i, bone_name in enumerate(all_names[:30]):  # Show first 30
            display_name = bone_name if bone_name else "(root)"
            is_core = "CORE" if bone_name in core_bone_names else ""
            print(f"  {i+1:2d}. {display_name:<20} {is_core}")
        
        if len(all_names) > 30:
            print(f"  ... and {len(all_names)-30} more bones")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during bone name analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run bone name analysis."""
    print("=" * 80)
    print("VMD Bone Name UTF Decoding Analysis")
    print("=" * 80)
    
    success = analyze_vmd_bone_names()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 Bone name analysis completed!")
    else:
        print("❌ Bone name analysis failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)