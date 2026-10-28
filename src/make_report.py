"""Converte relatorio/relatorio.md em DOCX (OOXML escrito à mão, sem dependências)
e em PDF (HTML + Microsoft Edge em modo headless).

Uso:  python -m src.make_report
Saídas: relatorio/Relatorio_APS_CNN.docx, relatorio/Relatorio_APS_CNN.pdf, relatorio/relatorio.html
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from markdown_it import MarkdownIt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "relatorio"
SRC = REP / "relatorio.md"
OUT_NAME = "Relatorio_APS_CNN"
EDGE = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")]

md = MarkdownIt("commonmark").enable("table")

# A4, margens de 2,5 cm -> largura útil 16 cm
PAGE_W, PAGE_H, MARGIN = 11906, 16838, 1418
CONTENT_DXA = PAGE_W - 2 * MARGIN
EMU_PER_DXA = 635


# =========================================================================== #
# DOCX
# =========================================================================== #
def run_xml(text, bold=False, italic=False, code=False, size=None, color=None):
    if not text:
        return ""
    rpr = ""  # ordem exigida pelo schema: rFonts, b, i, color, sz
    if code:
        rpr += '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
    if bold:
        rpr += "<w:b/>"
    if italic:
        rpr += "<w:i/>"
    if color:
        rpr += f'<w:color w:val="{color}"/>'
    if size or code:
        rpr += f'<w:sz w:val="{size or 19}"/>'
    rpr = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def inline_runs(children, base_bold=False):
    out, bold, italic = [], base_bold, False
    for t in children or []:
        if t.type == "text":
            out.append(run_xml(t.content, bold, italic))
        elif t.type == "code_inline":
            out.append(run_xml(t.content, bold, italic, code=True))
        elif t.type == "strong_open":
            bold = True
        elif t.type == "strong_close":
            bold = base_bold
        elif t.type == "em_open":
            italic = True
        elif t.type == "em_close":
            italic = False
        elif t.type in ("softbreak", "hardbreak"):
            out.append(run_xml(" "))
        elif t.type == "link_open":
            pass
    return "".join(out)


def para(content, style=None, align=None, num=None, keep_next=False, spacing=None):
    ppr = ""
    if style:
        ppr += f'<w:pStyle w:val="{style}"/>'
    if keep_next:
        ppr += "<w:keepNext/>"
    if num:
        ppr += f'<w:numPr><w:ilvl w:val="{num[1]}"/><w:numId w:val="{num[0]}"/></w:numPr>'
    if spacing:
        ppr += spacing
    if align:
        ppr += f'<w:jc w:val="{align}"/>'
    return f"<w:p><w:pPr>{ppr}</w:pPr>{content}</w:p>"


class Docx:
    def __init__(self):
        self.body, self.images, self.n_fig, self.n_tab = [], [], 0, 0
        self.nums = []  # (numId, abstractId)

    def new_num(self, ordered):
        nid = len(self.nums) + 1
        self.nums.append((nid, 1 if ordered else 0))
        return nid

    def image(self, path: Path, caption: str):
        self.n_fig += 1
        rid = f"rIdImg{len(self.images) + 1}"
        self.images.append((rid, path))
        w, h = Image.open(path).size
        max_w = CONTENT_DXA * EMU_PER_DXA
        max_h = int(max_w * 1.15)
        cx = max_w; cy = int(cx * h / w)
        if cy > max_h:
            cy = max_h; cx = int(cy * w / h)
        i = len(self.images)
        drawing = (
            f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{i}" name="Figura {i}"/>'
            f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="{i}" name="img{i}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
            f'</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')
        self.body.append(para(drawing, align="center", keep_next=True))
        self.body.append(para(run_xml(f"Figura {self.n_fig} — ", bold=True) + run_xml(caption),
                              style="Caption", align="center"))

    def table(self, rows, caption=None):
        """rows: lista de (is_header, [xml dos runs de cada célula], [alinhamentos])."""
        ncol = max(len(r[1]) for r in rows)
        # largura proporcional ao maior texto da coluna (limitada)
        lens = [0] * ncol
        for _, cells, _ in rows:
            for j, c in enumerate(cells):
                txt = re.sub(r"<[^>]+>", "", c)
                lens[j] = max(lens[j], min(len(txt), 38))
        tot = sum(max(l, 4) for l in lens)
        widths = [int(CONTENT_DXA * max(l, 4) / tot) for l in lens]
        widths[-1] += CONTENT_DXA - sum(widths)
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        trs = []
        for is_h, cells, aligns in rows:
            tcs = []
            for j, c in enumerate(cells):
                shade = '<w:shd w:val="clear" w:color="auto" w:fill="E8EEF7"/>' if is_h else ""
                jc = aligns[j] if j < len(aligns) and aligns[j] else "left"
                tcs.append(
                    f'<w:tc><w:tcPr><w:tcW w:w="{widths[j]}" w:type="dxa"/>{shade}</w:tcPr>'
                    f'<w:p><w:pPr><w:pStyle w:val="TableText"/><w:jc w:val="{jc}"/></w:pPr>{c}</w:p></w:tc>')
            trpr = "<w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>" if is_h else "<w:trPr><w:cantSplit/></w:trPr>"
            trs.append(f"<w:tr>{trpr}{''.join(tcs)}</w:tr>")
        b = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="C3C2B7"/>'
        borders = "".join(b.format(x) for x in ("top", "left", "bottom", "right", "insideH", "insideV"))
        self.body.append(
            f'<w:tbl><w:tblPr><w:tblW w:w="{CONTENT_DXA}" w:type="dxa"/><w:jc w:val="center"/>'
            f'<w:tblBorders>{borders}</w:tblBorders>'
            f'<w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>'
            f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(trs)}</w:tbl>')
        self.body.append(para("", spacing='<w:spacing w:after="60"/>'))

    # ----------------------------------------------------------------------- #
    def save(self, path: Path, title: str):
        W = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
             'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
             'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"')
        sect = (f'<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter"/>'
                f'<w:pgSz w:w="{PAGE_W}" w:h="{PAGE_H}"/>'
                f'<w:pgMar w:top="{MARGIN}" w:right="{MARGIN}" w:bottom="{MARGIN}" w:left="{MARGIN}" '
                f'w:header="709" w:footer="709" w:gutter="0"/><w:titlePg/></w:sectPr>')
        document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    f'<w:document {W}><w:body>{"".join(self.body)}{sect}</w:body></w:document>')
        footer = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr {W}>'
                  '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
                  '<w:r><w:rPr><w:color w:val="898781"/><w:sz w:val="18"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
                  '<w:r><w:rPr><w:color w:val="898781"/><w:sz w:val="18"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
                  '<w:r><w:rPr><w:color w:val="898781"/><w:sz w:val="18"/></w:rPr><w:fldChar w:fldCharType="separate"/></w:r>'
                  '<w:r><w:rPr><w:color w:val="898781"/><w:sz w:val="18"/></w:rPr><w:t>1</w:t></w:r>'
                  '<w:r><w:rPr><w:color w:val="898781"/><w:sz w:val="18"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r>'
                  '</w:p></w:ftr>')
        rels = ['<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
                '<Relationship Id="rIdNum" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>',
                '<Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>']
        for k, (rid, _) in enumerate(self.images, 1):
            rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image{k}.png"/>')
        doc_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    + "".join(rels) + "</Relationships>")
        ctypes = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                  '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                  '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                  '<Default Extension="xml" ContentType="application/xml"/>'
                  '<Default Extension="png" ContentType="image/png"/>'
                  '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                  '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
                  '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>'
                  '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
                  '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                  '</Types>')
        root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                     '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                     '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                     '</Relationships>')
        core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
                f'<dc:title>{escape(title)}</dc:title><dc:language>pt-BR</dc:language></cp:coreProperties>')
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", ctypes)
            z.writestr("_rels/.rels", root_rels)
            z.writestr("docProps/core.xml", core)
            z.writestr("word/document.xml", document)
            z.writestr("word/_rels/document.xml.rels", doc_rels)
            z.writestr("word/styles.xml", STYLES)
            z.writestr("word/numbering.xml", self.numbering())
            z.writestr("word/footer1.xml", footer)
            for k, (_, p) in enumerate(self.images, 1):
                z.write(p, f"word/media/image{k}.png")

    def numbering(self):
        def abstract(aid, ordered):
            lv = []
            for il in range(3):
                fmt, txt = ("decimal", f"%{il + 1}.") if ordered else ("bullet", "•–◦"[il])
                font = "" if ordered else '<w:rPr><w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI"/></w:rPr>'
                lv.append(f'<w:lvl w:ilvl="{il}"><w:start w:val="1"/><w:numFmt w:val="{fmt}"/>'
                          f'<w:lvlText w:val="{txt}"/><w:lvlJc w:val="left"/>'
                          f'<w:pPr><w:ind w:left="{360 * (il + 1) + 200}" w:hanging="300"/></w:pPr>{font}</w:lvl>')
            return f'<w:abstractNum w:abstractNumId="{aid}"><w:multiLevelType w:val="hybridMultilevel"/>{"".join(lv)}</w:abstractNum>'
        nums = "".join(f'<w:num w:numId="{n}"><w:abstractNumId w:val="{a}"/>'
                       f'<w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride></w:num>'
                       for n, a in self.nums)
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                + abstract(0, False) + abstract(1, True) + nums + "</w:numbering>")


STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Calibri" w:cs="Calibri"/>
<w:sz w:val="22"/><w:szCs w:val="22"/><w:lang w:val="pt-BR"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="288" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:pPr><w:jc w:val="both"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="1800" w:after="240"/><w:jc w:val="center"/></w:pPr><w:rPr><w:b/><w:color w:val="0D366B"/><w:sz w:val="44"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="120"/><w:jc w:val="center"/></w:pPr><w:rPr><w:color w:val="52514E"/><w:sz w:val="26"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="360" w:after="120"/><w:jc w:val="left"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:color w:val="0D366B"/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="80"/><w:jc w:val="left"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:color w:val="184F95"/><w:sz w:val="26"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="60"/><w:jc w:val="left"/><w:outlineLvl w:val="2"/></w:pPr><w:rPr><w:b/><w:color w:val="256ABF"/><w:sz w:val="23"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="caption"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="240"/><w:jc w:val="center"/></w:pPr><w:rPr><w:color w:val="52514E"/><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="30" w:after="30" w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="17"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:pPr><w:shd w:val="clear" w:color="auto" w:fill="F0EFEC"/><w:spacing w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="left"/></w:pPr><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="17"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Quote"><w:name w:val="Quote"/><w:basedOn w:val="Normal"/><w:pPr><w:pBdr><w:left w:val="single" w:sz="18" w:space="8" w:color="9EC5F4"/></w:pBdr><w:ind w:left="400"/></w:pPr><w:rPr><w:color w:val="52514E"/></w:rPr></w:style>
</w:styles>"""


def build_docx(tokens, out: Path, title: str):
    d = Docx()
    i, list_stack, quote = 0, [], False
    while i < len(tokens):
        t = tokens[i]
        if t.type == "heading_open":
            lvl = int(t.tag[1])
            inline = tokens[i + 1]
            style = {1: "Title", 2: "Heading1", 3: "Heading2", 4: "Heading3"}[lvl]
            d.body.append(para(inline_runs(inline.children), style=style))
            i += 3; continue
        if t.type == "blockquote_open":
            quote = True
        elif t.type == "blockquote_close":
            quote = False
        elif t.type in ("bullet_list_open", "ordered_list_open"):
            list_stack.append(d.new_num(t.type == "ordered_list_open"))
        elif t.type in ("bullet_list_close", "ordered_list_close"):
            list_stack.pop()
        elif t.type == "paragraph_open":
            inline = tokens[i + 1]
            ch = inline.children or []
            imgs = [c for c in ch if c.type == "image"]
            if imgs and all(c.type == "image" or (c.type == "text" and not c.content.strip())
                            for c in ch):
                for im in imgs:
                    d.image(ROOT / "relatorio" / im.attrs["src"], im.content)
            else:
                text = inline.content
                if text.startswith("{{subtitle}}"):
                    inline.children[0].content = inline.children[0].content.replace("{{subtitle}}", "")
                    d.body.append(para(inline_runs(inline.children), style="Subtitle"))
                elif text.strip() == "{{pagebreak}}":
                    d.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
                elif list_stack:
                    lvl = len(list_stack) - 1
                    d.body.append(para(inline_runs(ch), num=(list_stack[-1], min(lvl, 2)),
                                       spacing='<w:spacing w:after="60"/>'))
                else:
                    d.body.append(para(inline_runs(ch), style="Quote" if quote else None))
            i += 3; continue
        elif t.type == "fence" or t.type == "code_block":
            for line in t.content.rstrip("\n").split("\n"):
                d.body.append(para(run_xml(line or " "), style="Code"))
            d.body.append(para(""))
        elif t.type == "table_open":
            rows, j, is_h, cells, aligns = [], i + 1, False, [], []
            while tokens[j].type != "table_close":
                tj = tokens[j]
                if tj.type == "thead_open":
                    is_h = True
                elif tj.type == "tbody_open":
                    is_h = False
                elif tj.type == "tr_open":
                    cells, aligns = [], []
                elif tj.type in ("th_open", "td_open"):
                    st = tj.attrs.get("style", "") if tj.attrs else ""
                    aligns.append("right" if "right" in st else "center" if "center" in st else "left")
                elif tj.type == "inline":
                    cells.append(inline_runs(tj.children, base_bold=is_h))
                elif tj.type == "tr_close":
                    rows.append((is_h, cells, aligns))
                j += 1
            d.table(rows)
            i = j + 1; continue
        elif t.type == "hr":
            d.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
        i += 1
    d.save(out, title)
    return d.n_fig


# =========================================================================== #
# HTML / PDF
# =========================================================================== #
CSS = """
@page { size: A4; margin: 22mm 20mm 20mm 20mm;
        @bottom-center { content: counter(page); font: 9pt "Segoe UI", sans-serif; color: #898781; } }
@page :first { @bottom-center { content: none; } }
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --accent:#0d366b; --accent2:#184f95;
        --rule:#e1e0d9; --surface:#ffffff; --code:#f0efec; }
body { font-family: "Segoe UI", Calibri, system-ui, sans-serif; color: var(--ink);
       background: var(--surface); font-size: 10.5pt; line-height: 1.5; margin: 0; }
h1 { color: var(--accent); font-size: 24pt; text-align: center; margin: 60mm 0 6mm; line-height: 1.2; }
h2 { color: var(--accent); font-size: 15pt; margin: 9mm 0 2mm; border-bottom: 1px solid var(--rule);
     padding-bottom: 1mm; break-after: avoid; }
h3 { color: var(--accent2); font-size: 12pt; margin: 6mm 0 1.5mm; break-after: avoid; }
h4 { color: #256abf; font-size: 11pt; margin: 4mm 0 1mm; break-after: avoid; }
p { text-align: justify; margin: 0 0 2.5mm; }
p.subtitle { text-align: center; color: var(--ink2); font-size: 12.5pt; margin: 1.5mm 0; }
.fig { display: block; text-align: center; margin: 4mm 0 5mm; break-inside: avoid; }
.fig img { max-width: 100%; max-height: 205mm; }
.fig .cap { display: block; font-size: 8.8pt; color: var(--ink2); margin-top: 1.5mm; }
table { border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 2mm 0 5mm;
        break-inside: avoid; font-variant-numeric: tabular-nums; }
th, td { border: 1px solid #c3c2b7; padding: 1.2mm 1.6mm; }
th { background: #e8eef7; text-align: left; }
code { font-family: Consolas, monospace; font-size: 9pt; background: var(--code); padding: 0 1mm; }
pre { background: var(--code); padding: 3mm; font-size: 8.5pt; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { margin: 3mm 0; padding: 1mm 0 1mm 4mm; border-left: 3px solid #9ec5f4; color: var(--ink2); }
ul, ol { margin: 0 0 3mm; padding-left: 7mm; } li { margin-bottom: 1mm; }
.pb { break-after: page; }
"""


def build_html(text: str) -> str:
    n = {"fig": 0}

    def render_image(self, tokens, idx, options, env):
        n["fig"] += 1
        t = tokens[idx]
        return (f'<span class="fig"><img src="{html.escape(t.attrs["src"])}" alt="">'
                f'<span class="cap"><b>Figura {n["fig"]} —</b> {html.escape(t.content)}</span></span>')

    r = MarkdownIt("commonmark").enable("table")
    r.add_render_rule("image", render_image)
    body = r.render(text)
    body = re.sub(r"<p>\{\{subtitle\}\}(.*?)</p>", r'<p class="subtitle">\1</p>', body)
    body = body.replace("<p>{{pagebreak}}</p>", '<div class="pb"></div>').replace("<hr />", '<div class="pb"></div>')
    title = re.search(r"<h1>(.*?)</h1>", body)
    return (f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
            f'<title>{title.group(1) if title else "Relatório"}</title><style>{CSS}</style></head>'
            f"<body>{body}</body></html>")


def html_to_pdf(html_path: Path, pdf_path: Path):
    edge = next((p for p in EDGE if p.exists()), None)
    if edge is None:
        print("Edge não encontrado; PDF não gerado.")
        return False
    with tempfile.TemporaryDirectory() as prof:
        cmd = [str(edge), "--headless", "--disable-gpu", "--no-first-run",
               f"--user-data-dir={prof}", "--no-pdf-header-footer",
               "--print-to-pdf-no-header", f"--print-to-pdf={pdf_path}",
               html_path.resolve().as_uri()]
        subprocess.run(cmd, check=False, timeout=180, capture_output=True)
    return pdf_path.exists()


def main():
    text = SRC.read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", text, re.M).group(1)
    tokens = md.parse(text)
    n = build_docx(tokens, REP / f"{OUT_NAME}.docx", title)
    html_path = REP / "relatorio.html"
    html_path.write_text(build_html(text), encoding="utf-8")
    pdf = REP / f"{OUT_NAME}.pdf"
    if pdf.exists():
        pdf.unlink()
    ok = html_to_pdf(html_path, pdf)
    print(f"DOCX ok ({n} figuras); PDF {'ok' if ok else 'falhou'}")


if __name__ == "__main__":
    main()
