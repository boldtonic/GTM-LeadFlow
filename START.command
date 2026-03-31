#!/bin/bash
cd "$(dirname "$0")"
echo "================================================"
echo "  Fogata"
echo "================================================"
echo ""
echo "Starting server..."
echo "Open http://localhost:5001 in your browser"
echo ""
python3 app.py
