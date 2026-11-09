# Presentation: implementation and computational study

Beamer slides (English, IST visual identity) presenting the implementation of the
formulations M1, M2 and M3 of Talla Nobibon, Leus and Spieksma (2011) and the two
experiments stored in `results/`.

## Structure

```
presentation/
├── main.tex                 slides (19 numbered slides + cover, outline, thank-you, reference)
├── main.pdf                 compiled slides
├── figures/                 figures generated from the CSV files, and summary.txt with every number used
├── scripts/make_figures.py  builds figures/ from results/*.csv and instances/M3_small_extra/*.json
└── logo-tecnico.jpg         (add it here: the IST logo used in the title bar)
```

## Build

From the repository root:

```powershell
python presentation/scripts/make_figures.py
cd presentation
xelatex main.tex
xelatex main.tex
```

Run `xelatex` twice: the blue side band and the thank-you slide use TikZ overlays.
The font is Source Sans 3 when installed; otherwise the default sans-serif font is used.
`logo-tecnico.jpg` is optional: without it the slides compile without the logo.

## Colour convention

Slate text and blocks reproduce content of the paper. Tecnico Blue boxes, and slides with a
blue badge, blue left band and light blue background, contain the presenter's implementation,
experiments and interpretation.
