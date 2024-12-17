#!/bin/bash

# Display help message
usage() {
    echo "Usage: $0 <account> <sys> <path> [exclude]"
    echo
    echo "Description:"
    echo "  - <account>: User account name for login."
    echo "  - <sys>: System name (e.g., cluster or server name)."
    echo "  - <path>: Path to sync from the remote system (absolute or relative)."
    echo "  - [exclude]: Optional pattern to exclude files during sync."
    echo
    echo "Options:"
    echo "  -h, --help     Display this help message."
    echo
    exit 1
}

# Check if help is requested
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
fi

# Ensure correct number of arguments are passed
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    usage
fi

account="$1"
sys="$2"
path="$3"
exclude="$4"

# Ensure the path starts with a slash if it's meant to be an absolute path
if [[ "$path" != /* ]]; then
    path="/$path"
fi

# If no exclude argument is provided, set an empty string for exclude
if [ -z "$exclude" ]; then
    exclude=""
fi

# Perform the rsync operation and exit after completion
if [ -n "$exclude" ]; then
    rsync -avz --exclude="$exclude" "$account@login.$sys.cineca.it:$path" ./
else
    rsync -avz "$account@login.$sys.cineca.it:$path" ./
fi

# Exit the script after the rsync operation completes
exit 0

