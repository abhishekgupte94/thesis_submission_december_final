# ast_sanity_check.py - Test script for debugging AST audio extraction

import torch
import sys
from pathlib import Path

# Add your project root to path
sys.path.insert(0, '/lambda/nfs/projectxfilesystem/thesis_main_project_final_submission_in_april')

from thesis_main_files.main_files.feature_extraction.new_file_setups.Audio_Feature_Extraction.ast_huggingface.extract_audio_features_from_AST import \
    ASTAudioExtractor


def test_ast_extractor():
    print("🧪 AST Audio Extractor Sanity Check")
    print("=" * 50)

    # Test paths (use the ones that were failing)
    test_paths = [
        '/lambda/nfs/projectxfilesystem/thesis_main_project_final_submission_in_april/thesis_main_files/datasets/processed/lav_df/new_setup/train_files/sample_real_70_percent_half1/074838.mp4',
        '/lambda/nfs/projectxfilesystem/thesis_main_project_final_submission_in_april/thesis_main_files/datasets/processed/lav_df/new_setup/train_files/sample_real_70_percent_half1/098144.mp4',
        '/lambda/nfs/projectxfilesystem/thesis_main_project_final_submission_in_april/thesis_main_files/datasets/processed/lav_df/new_setup/train_files/sample_real_70_percent_half1/055888.mp4'
    ]

    # Check if files exist
    print("📁 File existence check:")
    for i, path in enumerate(test_paths):
        exists = Path(path).exists()
        size = Path(path).stat().st_size if exists else 0
        print(f"  {i + 1}. {Path(path).name}: {'✅' if exists else '❌'} ({size} bytes)")

    # Test 1: Initialize AST extractor
    print("\n🔧 Initializing AST extractor...")
    try:
        extractor = ASTAudioExtractor(
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True
        )
        print(f"✅ AST extractor initialized on device: {extractor.device}")
    except Exception as e:
        print(f"❌ Failed to initialize AST extractor: {e}")
        return

    # Test 2: Extract from single file
    print(f"\n🎵 Testing single file extraction...")
    try:
        single_result = extractor.extract_one(test_paths[0])
        if single_result is not None:
            print(f"✅ Single extraction success:")
            print(f"   Path: {single_result.get('path', 'N/A')}")
            print(f"   Shape: {single_result.get('shape', 'N/A')}")
            if 'features' in single_result:
                feat = single_result['features']
                print(f"   Features type: {type(feat)}")
                if torch.is_tensor(feat):
                    print(f"   Features shape: {tuple(feat.shape)}")
                    print(f"   Features device: {feat.device}")
                    print(f"   Features dtype: {feat.dtype}")
                    print(f"   Features range: [{feat.min().item():.4f}, {feat.max().item():.4f}]")
                else:
                    print(f"   ❌ Features not a tensor: {feat}")
        else:
            print(f"❌ Single extraction returned None")
    except Exception as e:
        print(f"❌ Single extraction failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Extract from multiple files (the actual failing case)
    print(f"\n🎵 Testing batch extraction (extract_from_paths)...")
    try:
        batch_results = extractor.extract_from_paths(test_paths[:2])  # Test with 2 files
        print(f"📊 Batch extraction results:")
        print(f"   Input paths: {len(test_paths[:2])}")
        print(f"   Output results: {len(batch_results)}")

        if batch_results:
            for i, result in enumerate(batch_results):
                if result is not None:
                    print(f"   Result {i}: {result.get('shape', 'no shape')}")
                    if 'features' in result and torch.is_tensor(result['features']):
                        print(f"     Features shape: {tuple(result['features'].shape)}")
                else:
                    print(f"   Result {i}: None")
        else:
            print(f"   ❌ No results returned")

        # Test the problematic lines from your code
        print(f"\n🔍 Testing the failing code path:")
        items = batch_results
        print(f"   items length: {len(items)}")

        feats = [it["features"] for it in items if it is not None]
        print(f"   feats length: {len(feats)}")
        print(f"   feats types: {[type(f) for f in feats]}")

        if feats:
            valid_feats = [f for f in feats if f is not None and torch.is_tensor(f) and f.numel() > 0]
            print(f"   valid_feats length: {len(valid_feats)}")

            if valid_feats:
                shapes = {tuple(f.shape) for f in valid_feats}
                print(f"   shapes: {shapes}")

                if len(shapes) == 1:
                    print(f"   ✅ All shapes match, can stack")
                    try:
                        stacked = torch.stack(valid_feats, dim=0)
                        print(f"   ✅ Stack successful: {tuple(stacked.shape)}")
                    except Exception as e:
                        print(f"   ❌ Stack failed: {e}")
                else:
                    print(f"   ❌ Shape mismatch: {shapes}")
            else:
                print(f"   ❌ No valid features")
        else:
            print(f"   ❌ No features extracted")

    except Exception as e:
        print(f"❌ Batch extraction failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Check audio in the video files
    print(f"\n🔊 Checking audio streams in video files...")
    import subprocess
    for i, path in enumerate(test_paths[:2]):
        try:
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_name', '-of',
                   'csv=p=0', str(path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                print(f"   File {i + 1}: Audio codec = {result.stdout.strip()}")
            else:
                print(f"   File {i + 1}: ❌ No audio stream found")
        except Exception as e:
            print(f"   File {i + 1}: Error checking audio: {e}")


if __name__ == "__main__":
    test_ast_extractor()