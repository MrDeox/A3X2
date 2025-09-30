# Secure patch.py script with safety checks
import sys
import os
import subprocess
import tempfile
import shutil

# Safety configurations
ALLOWED_DIRS = ['/home/arthur/Projetos/A3X']  # Restrict patching to allowed directories
MAX_FILE_SIZE = 1024 * 1024  # 1MB max file size


def validate_path(path):
    """Validate that the path is safe to operate on."""
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    for allowed_dir in ALLOWED_DIRS:
        if path.startswith(allowed_dir):
            return path
    raise ValueError(f"Path {path} is not in allowed directories")


def validate_diff_content(diff_content):
    """Basic validation of diff content to prevent malicious patches."""
    if len(diff_content) > 10000:  # Limit diff size
        raise ValueError("Diff too large")
    if 'rm -rf' in diff_content or 'exec' in diff_content.lower():
        raise ValueError("Potentially malicious diff content detected")
    return True


def apply_secure_patch(file_path, diff_content):
    """Apply patch with security checks."""
    # Validate inputs
    validate_path(file_path)
    validate_diff_content(diff_content)

    # Check file size
    if os.path.exists(file_path) and os.path.getsize(file_path) > MAX_FILE_SIZE:
        raise ValueError("File too large to patch")

    # Create backup
    backup_path = file_path + '.backup'
    if os.path.exists(file_path):
        shutil.copy2(file_path, backup_path)

    try:
        # Write diff to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(diff_content)
            temp_patch = f.name

        # Run patch command with restrictions
        cmd = ['patch', '-p0', '-i', temp_patch, file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise RuntimeError(f"Patch failed: {result.stderr}")

        # Clean up temp file
        os.unlink(temp_patch)

        print(f"Patch applied successfully to {file_path}")

    except Exception as e:
        # Restore backup on failure
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)
        raise e

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python patch.py <file_path> <diff_content>")
        sys.exit(1)

    file_path = sys.argv[1]
    diff_content = ' '.join(sys.argv[2:])

    try:
        apply_secure_patch(file_path, diff_content)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
