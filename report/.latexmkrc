# Cấu hình cho latexmk (VS Code LaTeX Workshop / terminal): bật shell-escape cho gói minted
$pdf_mode = 1;
$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode -synctex=1 %O %S';
