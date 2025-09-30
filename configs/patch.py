#!/usr/bin/env python3
import sys
import os
import subprocess
import shutil
import re

# Security checks added
ALLOWED_DIRS = ['/home', '/tmp']  # Restrict patching to safe directories
def is_safe_path(path):
    for allowed in ALLOWED_DIRS:
        if path.startswith(allowed):
            return True
    return False

def validate_patch_content(patch_content):
    # Basic check for dangerous commands in patch
    dangerous_patterns = [r'rm\s+-rf', r'/bin/sh', r'exec\s+', r'sudo\s+']
    for pattern in dangerous_patterns:
        if re.search(pattern, patch_content, re.IGNORECASE):
            return False, f'Dangerous pattern found: {pattern}'
    return True, 'Safe'

def main():
    if len(sys.argv) != 3:
        print('Usage: python patch.py <patchfile> <targetfile>')
        sys.exit(1)

    patchfile = sys.argv[1]
    targetfile = sys.argv[2]

    # Security check 1: Validate paths
    if not os.path.exists(patchfile):
        print(f'Error: Patch file {patchfile} does not exist.')
        sys.exit(1)
    if not is_safe_path(os.path.abspath(targetfile)):
        print(f'Error: Target file {targetfile} is not in allowed directory.')
        sys.exit(1)

    # Security check 2: Backup original if exists
    if os.path.exists(targetfile):
        backup = targetfile + '.bak'
        if os.path.exists(backup):
            print(f'Warning: Backup {backup} already exists. Overwriting.')
        shutil.copy2(targetfile, backup)
        print(f'Backup created: {backup}')

    # Security check 3: Validate patch content
    with open(patchfile, 'r') as f:
        patch_content = f.read()
    is_safe, msg = validate_patch_content(patch_content)
    if not is_safe:
        print(f'Security violation: {msg}')
        sys.exit(1)
    print(f'Patch validation: {msg}')

    # Apply patch
    try:
        result = subprocess.run(['patch', '-i', patchfile, targetfile], capture_output=True, text=True)
        if result.returncode == 0:
            print('Patch applied successfully.')
        else:
            print(f'Patch failed: {result.stderr}')
            sys.exit(1)
    except Exception as e:
        print(f'Error applying patch: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()