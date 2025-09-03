import os
import h5py
import numpy as np
import re
import xml.etree.ElementTree as ET
from glob import glob

# Define directory paths
urdf_dir = "/home/ran.ding/projects/TARGO/data//acronym/urdfs_acronym"
h5_dir = "/usr/stud/dira/GraspInClutter/acronym/data/grasps"
backup_dir = "/home/ran.ding/projects/TARGO/data//acronym/urdfs_acronym"

# Ensure backup directory exists
os.makedirs(backup_dir, exist_ok=True)

# Statistics variables
total_files = 0
updated_files = 0
no_h5_files = 0
errors = 0

print(f"Starting URDF file processing...")

# First get all h5 file paths and index by object ID
h5_files_dict = {}
for h5_file in glob(os.path.join(h5_dir, "*.h5")):
    h5_basename = os.path.basename(h5_file)
    # Look for both exact ID match and ID contained in filename
    match = re.match(r"([^_]+)_([^_]+)_([^.]+)\.h5", h5_basename)
    if match:
        category, obj_id, scale = match.groups()
        if obj_id not in h5_files_dict:
            h5_files_dict[obj_id] = []
        h5_files_dict[obj_id].append(h5_file)
    
    # Also index by full object name (e.g., Pen_e729f6bff8f325152daa89e31f28255d)
    # This helps find h5 files when only the full object name is available
    h5_name_without_ext = os.path.splitext(h5_basename)[0]
    parts = h5_name_without_ext.split('_')
    if len(parts) >= 2:
        full_obj_name = f"{parts[0]}_{parts[1]}"
        if full_obj_name not in h5_files_dict:
            h5_files_dict[full_obj_name] = []
        h5_files_dict[full_obj_name].append(h5_file)

print(f"Found H5 files for {len(h5_files_dict)} different object IDs")

# Iterate through URDF files
for urdf_file in os.listdir(urdf_dir):
    if not urdf_file.endswith(".urdf"):
        continue
    
    total_files += 1
    urdf_path = os.path.join(urdf_dir, urdf_file)
    
    try:
        # Parse URDF filename
        match = re.match(r"([^_]+)_([^.]+)\.urdf", urdf_file)
        if not match:
            print(f"Warning: Filename {urdf_file} does not match expected format")
            continue
            
        category, obj_id = match.groups()
        full_obj_name = f"{category}_{obj_id}"
        
        # Try to find corresponding H5 file - first by ID, then by full name
        h5_files = []
        if obj_id in h5_files_dict:
            h5_files = h5_files_dict[obj_id]
        elif full_obj_name in h5_files_dict:
            h5_files = h5_files_dict[full_obj_name]
            
        if not h5_files:
            print(f"警告: 未找到物体ID '{obj_id}'对应的H5文件，跳过 {urdf_file}")
            no_h5_files += 1
            continue
            
        # Use the first matching H5 file
        h5_file = h5_files[0]
        print(f"Processing: {urdf_file} (using H5 file: {os.path.basename(h5_file)})")
        
        # Read physical properties from H5 file
        with h5py.File(h5_file, 'r') as f:
            # Read mass
            if 'object' in f and 'mass' in f['object']:
                mass = f['object']['mass'][()]
            else:
                print(f"Warning: Mass data not found in H5 file")
                mass = None
            
            # Read inertia matrix
            if 'object' in f and 'inertia' in f['object']:
                inertia = f['object']['inertia'][()]
            else:
                print(f"Warning: Inertia matrix data not found in H5 file")
                inertia = None
            
            # Read friction coefficient
            if 'object' in f and 'friction' in f['object']:
                friction = f['object']['friction'][()]
            else:
                print(f"Warning: Friction coefficient data not found in H5 file")
                friction = None
                
            # Read center of mass
            if 'object' in f and 'com' in f['object']:
                com = f['object']['com'][()]
            else:
                print(f"Warning: Center of mass data not found in H5 file")
                com = None
        
        # Backup original URDF file
        backup_path = os.path.join(backup_dir, urdf_file)
        if not os.path.exists(backup_path):
            with open(urdf_path, 'r') as f:
                original_content = f.read()
            with open(backup_path, 'w') as f:
                f.write(original_content)
        
        # Parse URDF file
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        # Update mass
        if mass is not None:
            for inertial in root.findall(".//inertial"):
                for mass_elem in inertial.findall("mass"):
                    mass_elem.set("value", str(mass))
        
        # Update inertia matrix
        if inertia is not None:
            for inertial in root.findall(".//inertial"):
                for inertia_elem in inertial.findall("inertia"):
                    inertia_elem.set("ixx", str(inertia[0][0]))
                    inertia_elem.set("ixy", str(inertia[0][1]))
                    inertia_elem.set("ixz", str(inertia[0][2]))
                    inertia_elem.set("iyy", str(inertia[1][1]))
                    inertia_elem.set("iyz", str(inertia[1][2]))
                    inertia_elem.set("izz", str(inertia[2][2]))
        
        # Update friction coefficient
        if friction is not None:
            for contact in root.findall(".//contact"):
                for friction_elem in contact.findall("lateral_friction"):
                    friction_elem.set("value", str(friction))
        
        # Add center of mass information (if not present)
        if com is not None:
            for inertial in root.findall(".//inertial"):
                # Check if origin element already exists
                origin_elem = inertial.find("origin")
                if origin_elem is None:
                    # Create new origin element
                    origin_elem = ET.SubElement(inertial, "origin")
                    origin_elem.set("xyz", f"{com[0]} {com[1]} {com[2]}")
                    origin_elem.set("rpy", "0 0 0")
                else:
                    # Update existing origin element
                    origin_elem.set("xyz", f"{com[0]} {com[1]} {com[2]}")
        
        # Save updated URDF file
        tree.write(urdf_path)
        updated_files += 1
        
        # Output progress every 10 files
        if total_files % 10 == 0:
            print(f"Processed {total_files} files, updated {updated_files} files")
        
    except Exception as e:
        errors += 1
        print(f"Error processing file {urdf_file}: {str(e)}")

# Output statistics
print("\nProcessing complete!")
print(f"Total URDF files: {total_files}")
print(f"Updated files: {updated_files}")
print(f"URDF files without H5 data: {no_h5_files}")
print(f"Files with processing errors: {errors}")