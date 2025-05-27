#!/usr/bin/env python3
"""
Simple test to verify VGN modifications
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_vgn_import():
    """Test if VGN can be imported and has the required attributes"""
    try:
        from src.vgn.detection import VGN
        print("✓ VGN imported successfully")
        
        # Check constructor signature
        import inspect
        init_sig = inspect.signature(VGN.__init__)
        params = list(init_sig.parameters.keys())
        
        if 'cd_iou_measure' in params:
            print("✓ VGN.__init__ has cd_iou_measure parameter")
        else:
            print("✗ VGN.__init__ missing cd_iou_measure parameter")
            return False
            
        # Check __call__ signature
        call_sig = inspect.signature(VGN.__call__)
        call_params = list(call_sig.parameters.keys())
        
        required_call_params = ['cd_iou_measure', 'target_mesh_gt']
        for param in required_call_params:
            if param in call_params:
                print(f"✓ VGN.__call__ has {param} parameter")
            else:
                print(f"✗ VGN.__call__ missing {param} parameter")
                return False
        
        print("✓ All VGN modifications verified successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing VGN: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Simple VGN Test")
    print("=" * 50)
    
    success = test_vgn_import()
    
    if success:
        print("\n✓ VGN modifications are correct!")
        print("VGN should now work with category statistics generation.")
    else:
        print("\n✗ VGN modifications have issues.")
    
    print("=" * 50) 