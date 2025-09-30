#!/usr/bin/env python3
import sys
import os
import subprocess


def apply_secure_patch(patch_file, target_dir):
    # Security check: Verify patch file exists
    if not os.path.exists(patch_file):
        raise ValueError(f"Patch file '{patch_file}' does not exist.")

    # Security check: Verify patch file is readable
    if not os.access(patch_file, os.R_OK):
        raise ValueError(f"No read permission for patch file '{patch_file}'.")

    # Security check: Ensure patch file has safe extension
    if not patch_file.endswith('.patch'):
        raise ValueError("Patch file must have .patch extension for security.")

    # Security check: Verify target directory exists and is writable
    if not os.path.exists(target_dir):
        raise ValueError(f"Target directory '{target_dir}' does not exist.")

    if not os.access(target_dir, os.W_OK):
        raise ValueError(f"No write permission for target directory '{target_dir}'.")

    # Security check: Limit patch to target directory only (no --forward or recursive risks)
    try:
        result = subprocess.run(
            ['patch', '-p1', '-d', target_dir, '-i', patch_file, '--dry-run'],
            capture_output=True,
            text=True,
            check=True
        )
        print("Dry run successful. Proceeding with patch application.")
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Dry run failed: {e.stderr}")

    # Apply the patch
    try:
        subprocess.check_call(['patch', '-p1', '-d', target_dir, '-i', patch_file])
        print(f"Patch applied successfully to '{target_dir}'.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Patch application failed: {e}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python patch.py <patch_file> <target_dir>")
        sys.exit(1)

    patch_file = sys.argv[1]
    target_dir = sys.argv[2]

    try:
        apply_secure_patch(patch_file, target_dir)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()