#!/bin/bash

# LaTeX Environment Test Script
# This script verifies that all necessary LaTeX tools and packages are available

echo "🔍 Testing LaTeX environment..."
echo "================================"

# Test basic commands
echo "📝 Checking basic LaTeX tools:"
commands=("pdflatex" "bibtex" "latexmk" "tlmgr")

for cmd in "${commands[@]}"; do
    if command -v "$cmd" &> /dev/null; then
        echo "  ✅ $cmd: $(command -v $cmd)"
    else
        echo "  ❌ $cmd: not found"
        exit 1
    fi
done

echo ""
echo "📦 Checking TeX Live version:"
pdflatex --version | head -1

echo ""
echo "📚 Testing package availability:"
packages=("ieeetran" "cite" "natbib" "hyperref" "graphicx" "amsmath" "tikz" "pgfplots")

for pkg in "${packages[@]}"; do
    if kpsewhich "${pkg}.sty" &> /dev/null || kpsewhich "${pkg}.cls" &> /dev/null; then
        echo "  ✅ $pkg"
    else
        echo "  ❌ $pkg: not found"
    fi
done

echo ""
echo "🏗️  Testing compilation with a minimal document:"

# Create a minimal test document
cat > /tmp/test.tex << 'EOF'
\documentclass[conference]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}

\begin{document}
\title{Test Document}
\author{LaTeX Test}
\maketitle

\section{Introduction}
This is a test document to verify LaTeX compilation.

\begin{equation}
E = mc^2
\end{equation}

\begin{thebibliography}{1}
\bibitem{test}
Test Author, ``Test Title,'' \emph{Test Journal}, 2024.
\end{thebibliography}

\end{document}
EOF

# Test compilation
cd /tmp
if pdflatex -interaction=nonstopmode test.tex > /dev/null 2>&1; then
    echo "  ✅ Basic compilation successful"
    if [ -f "test.pdf" ]; then
        echo "  ✅ PDF generated successfully"
        rm -f test.*
    else
        echo "  ❌ PDF not generated"
    fi
else
    echo "  ❌ Compilation failed"
    echo "Check the log for details:"
    cat test.log
fi

echo ""
echo "🎉 LaTeX environment test completed!"
echo ""
echo "💡 To compile your paper, run:"
echo "   cd /workspace/latex && latexmk -pdf main.tex"