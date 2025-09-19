# Use Debian bookworm as base image for consistency with the current devcontainer
FROM debian:bookworm-slim

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies and basic tools
RUN apt-get update && apt-get install -y \
    # Essential system tools
    curl \
    wget \
    git \
    vim \
    nano \
    unzip \
    zip \
    # Build tools that might be needed
    build-essential \
    # Python tools (for potential automation scripts)
    python3 \
    python3-pip \
    # Clean up to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install TeX Live full distribution
# We use the network installer to get the most up-to-date version
RUN mkdir -p /tmp/texlive && cd /tmp/texlive \
    && wget -O install-tl-unx.tar.gz http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz \
    && tar -xzf install-tl-unx.tar.gz \
    && cd install-tl-* \
    # Create a profile for automated installation
    && echo "selected_scheme scheme-full" > texlive.profile \
    && echo "tlpdbopt_install_docfiles 0" >> texlive.profile \
    && echo "tlpdbopt_install_srcfiles 0" >> texlive.profile \
    && echo "tlpdbopt_autobackup 0" >> texlive.profile \
    && echo "instopt_adjustpath 1" >> texlive.profile \
    # Install TeX Live
    && ./install-tl --profile=texlive.profile \
    # Clean up installation files
    && cd / && rm -rf /tmp/texlive

# Add TeX Live to PATH
ENV PATH="/usr/local/texlive/2025/bin/x86_64-linux:${PATH}"

# Ensure font database is up to date
RUN texhash && updmap-sys

# Install additional LaTeX packages that we used during compilation
# (Most should already be included in scheme-full, but this ensures they're available)
RUN tlmgr install \
    cite \
    fancyvrb \
    ieeetran \
    paralist \
    csvsimple \
    titlesec \
    soul \
    markdown \
    adjustbox \
    booktabs \
    multirow \
    pgfplots \
    float \
    natbib \
    algorithms \
    times \
    psnfss \
    # Additional packages that are commonly needed
    xcolor \
    hyperref \
    graphicx \
    amsmath \
    amsfonts \
    amssymb \
    tikz

# Update font maps and rebuild formats
RUN updmap-sys && fmtutil-sys --all

# Create a non-root user for development
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid vscode --shell /bin/bash --create-home vscode

# Install latexmk configuration for better compilation
RUN echo 'use utf8;' > /home/vscode/.latexmkrc \
    && echo '$pdf_mode = 1;' >> /home/vscode/.latexmkrc \
    && echo '$bibtex_use = 2;' >> /home/vscode/.latexmkrc \
    && echo '$pdflatex = "pdflatex -synctex=1 -interaction=nonstopmode";' >> /home/vscode/.latexmkrc \
    && chown vscode:vscode /home/vscode/.latexmkrc

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER vscode

# Verify installation by checking key commands
RUN pdflatex --version && bibtex --version && latexmk --version

# Default command
CMD ["/bin/bash"]