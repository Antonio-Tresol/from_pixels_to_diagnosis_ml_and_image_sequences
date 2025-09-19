#!/bin/bash

# Quick LaTeX compilation script
# Usage: ./compile.sh [clean]

cd "$(dirname "$0")"

if [ "$1" = "clean" ]; then
    echo "🧹 Cleaning build files..."
    latexmk -C
    echo "✅ Clean completed"
    exit 0
fi

echo "📝 Compiling LaTeX document..."
echo "=============================="

if latexmk -pdf main.tex; then
    echo ""
    echo "🎉 Compilation successful!"
    echo "📄 Output: $(pwd)/main.pdf"
    
    # Show file size
    if [ -f "main.pdf" ]; then
        size=$(du -h main.pdf | cut -f1)
        echo "📊 File size: $size"
    fi
else
    echo ""
    echo "❌ Compilation failed!"
    echo "Check main.log for details"
    exit 1
fi