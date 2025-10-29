#!/bin/bash
cd /home/kavia/workspace/code-generation/image-processing-suite-37224-37233/image_processing_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

