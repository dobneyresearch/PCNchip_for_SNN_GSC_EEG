#!/usr/bin/env python3
"""Assemble main_stage3_GSC_EEG_v1.tex from the v1.md (prose + raw-LaTeX equations)
+ the shared v3.tex preamble/macros. Equations are written directly in the .md as
raw \\begin{equation} environments (pandoc passes them through), so there is NO
equation/figure splicing here; only the title/author are overridden and the body is
pandoc-converted."""
import re, subprocess, pathlib

HERE = pathlib.Path(__file__).parent
tex3 = (HERE / "main_stage2_v3.tex").read_text()
md = (HERE / "main_stage3_GSC_EEG_v1.md").read_text()

# ---- 1. preamble (everything up to \begin{document}); swap title/author/date ----
pre = tex3.split(r"\begin{document}")[0]
new_titleblock = (
    r"\title{\bfseries One Leaky-Accumulator Cell: A Generalist Forwards-Only Rule for\\"
    "\n"
    r"Temporal Credit Assignment Across Spiking Speech and EEG}"
    "\n"
    r"\author{Saul Dobney\thanks{Correspondence: \texttt{saul.dobney@dobney.com}."
    "\n"
    r"This is Part~III of the study; Part~I~\cite{dobney2026analog} presents the analog cell and its"
    "\n"
    r"unsupervised learning, and Part~II~\cite{dobney2026forwards} the forwards-only supervised"
    "\n"
    r"architecture. Code and reproduction scripts are available at"
    "\n"
    r"\url{https://github.com/dobneyresearch/PCNchip_with_leakyjug_learning}.}}"
    "\n"
    r"\date{2026}"
)
pre = re.sub(r"\\title\{.*?\\date\{2026\}", lambda _m: new_titleblock, pre, flags=re.S)

# ---- 2. pandoc table-support macros ----
tbl_pre = (r"\usepackage{longtable,booktabs,array}" "\n"
           r"\usepackage{calc}" "\n"
           r"\providecommand{\tightlist}{"
           r"\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}" "\n"
           r"\providecommand{\passthrough}[1]{#1}" "\n"
           r"\providecommand{\pandocbounded}[1]{#1}" "\n")

# ---- 3. pandoc helper ----
def pandoc(text):
    r = subprocess.run(["pandoc", "-f", "markdown", "-t", "latex", "--natbib"],
                       input=text, capture_output=True, text=True)
    out = r.stdout
    out = out.replace(r"\citep{", r"\cite{").replace(r"\citet{", r"\cite{")
    out = out.replace(r"\autocite{", r"\cite{")
    return out

# ---- 4. abstract (in YAML block) + body ----
ab = re.search(r"\nabstract:\s*\|\n(.*?)\nauthor:", md, re.S).group(1)
ab = "\n".join(l[2:] if l.startswith("  ") else l for l in ab.splitlines())
abstract_tex = pandoc(ab).strip()

body_md = md.split("\n---\n", 1)[1]
body_md = body_md.split("---", 1)[1] if body_md.lstrip().startswith("-") else body_md
body_md = body_md[body_md.index("\n# "):]
body = pandoc(body_md)

# tidy: drop any stray raw HTML comments pandoc passed through as verbatim
body = re.sub(r"\\begin\{verbatim\}<!--.*?-->\\end\{verbatim\}", "", body, flags=re.S)

# ---- 5. assemble ----
doc = (pre
       + tbl_pre
       + "\\begin{document}\n\\maketitle\n\n"
       + "\\begin{abstract}\n" + abstract_tex + "\n\\end{abstract}\n\n"
       + body
       + "\n\n\\bibliographystyle{IEEEtran}\n\\bibliography{refs}\n"
       + "\\end{document}\n")
# normalize exotic unicode spaces pandoc left in the prose
for cp in (0x2006,0x2009,0x2005,0x2007,0x2008,0x202F,0x2002,0x2003,0x00A0,0x2004,0x200A,0x2028):
    doc = doc.replace(chr(cp), " ")
doc = doc.replace(chr(0x2011), "-")
(HERE / "main_stage3_GSC_EEG_v1.tex").write_text(doc)
print("wrote main_stage3_GSC_EEG_v1.tex")
