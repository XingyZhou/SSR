"""Unpack the Rikkyo template, duplicate the inner page into slides 4-7, run the
weekly generator, and zip the result over the output pptx."""
import os, re, shutil, subprocess, sys, zipfile
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(REPO, "周报ppt/template_rikkyo_ai.pptx")
GEN = os.path.join(REPO, "周报ppt/gen_weekly_pptx.py")
OUT = os.path.join(REPO, "周报ppt/2026年第42週_進捗報告.pptx")
WORK = os.path.join(os.environ.get("TMPDIR", "/tmp"), "weekly_pptx_unpacked")

if os.path.isdir(WORK):
    shutil.rmtree(WORK)
with zipfile.ZipFile(TEMPLATE) as z:
    z.extractall(WORK)

SL = os.path.join(WORK, "ppt/slides")
SLIDE_CT = "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
SLIDE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"

src = open(os.path.join(SL, "slide2.xml"), encoding="utf-8").read()
rels = open(os.path.join(SL, "_rels/slide2.xml.rels"), encoding="utf-8").read()
rels = re.sub(r'<Relationship [^>]*notesSlide[^>]*/>', "", rels)  # a notes page cannot be shared

ct_path = os.path.join(WORK, "[Content_Types].xml")
ct = open(ct_path, encoding="utf-8").read()
prels_path = os.path.join(WORK, "ppt/_rels/presentation.xml.rels")
prels = open(prels_path, encoding="utf-8").read()
pres_path = os.path.join(WORK, "ppt/presentation.xml")
pres = open(pres_path, encoding="utf-8").read()

new_ids = []
for n, rid, sid in [(4, 11, 259), (5, 12, 260), (6, 13, 261), (7, 14, 262)]:
    open(os.path.join(SL, f"slide{n}.xml"), "w", encoding="utf-8").write(src)
    open(os.path.join(SL, f"_rels/slide{n}.xml.rels"), "w", encoding="utf-8").write(rels)
    ct = ct.replace("</Types>", f'<Override PartName="/ppt/slides/slide{n}.xml" ContentType="{SLIDE_CT}"/></Types>')
    prels = prels.replace("</Relationships>", f'<Relationship Id="rId{rid}" Type="{SLIDE_REL}" Target="slides/slide{n}.xml"/></Relationships>')
    new_ids.append(f'<p:sldId id="{sid}" r:id="rId{rid}"/>')
pres = pres.replace("</p:sldIdLst>", "".join(new_ids) + "</p:sldIdLst>")
open(ct_path, "w", encoding="utf-8").write(ct)
open(prels_path, "w", encoding="utf-8").write(prels)
open(pres_path, "w", encoding="utf-8").write(pres)

subprocess.run([sys.executable, GEN, WORK], check=True)

# every XML part must still parse
bad = []
for root, _, files in os.walk(WORK):
    for f in files:
        if f.endswith((".xml", ".rels")):
            try:
                ET.parse(os.path.join(root, f))
            except ET.ParseError as e:
                bad.append((os.path.join(root, f), str(e)))
if bad:
    for b in bad:
        print("XML ERROR", b)
    sys.exit(1)

tmp = OUT + ".tmp"
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(ct_path, "[Content_Types].xml")
    for root, _, files in os.walk(WORK):
        for f in files:
            p = os.path.join(root, f)
            arc = os.path.relpath(p, WORK)
            if arc == "[Content_Types].xml":
                continue
            z.write(p, arc)
os.replace(tmp, OUT)
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")
