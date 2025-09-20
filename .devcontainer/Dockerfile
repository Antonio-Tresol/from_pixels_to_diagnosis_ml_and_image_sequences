# FASTEST OPTION: Pre-built LaTeX image
FROM texlive/texlive:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git vim nano && apt-get clean

# Setup latexmk configuration in root (simpler approach)
RUN echo '$pdf_mode = 1; $bibtex_use = 2; $pdflatex = "pdflatex -synctex=1 -interaction=nonstopmode";' > /.latexmkrc

WORKDIR /workspace
CMD ["/bin/bash"]