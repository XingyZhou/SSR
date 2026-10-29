#!/usr/bin/env python3
"""Fill the Rikkyo AI weekly-report template (unpacked pptx) with the 2026-10-28 content
(V6 / V7 experiments, results and band diagnosis). Same helpers as gen_weekly_pptx.py plus
picture(): run via  python 周报ppt/build_weekly_pptx.py 周报ppt/gen_weekly_pptx_1028.py <out.pptx> 6"""
import re, sys, os, shutil
from xml.sax.saxutils import escape

ROOT = sys.argv[1]  # unpacked dir
HERE = os.path.dirname(os.path.abspath(__file__))
SL = os.path.join(ROOT, "ppt", "slides")
EMU = 914400
FONT = "IBM Plex Sans JP"
DARK = "1D2129"; GRAY = "6B6480"; MAG = "D6307F"; PUR = "4B2A8C"; MID = "8A2F8C"
TINT = "F6E6F0"; CARD = "F7F4FA"; LINE = "E3DCEA"; WHITE = "FFFFFF"; BODY = "3A3550"

_id = [10]
def nid():
    _id[0] += 1
    return _id[0]

def E(v):  # inches -> EMU
    return int(round(v * EMU))

def rpr(sz, b=False, color=DARK):
    bold = ' b="1"' if b else ''
    return (f'<a:rPr lang="ja-JP" altLang="en-US" sz="{int(sz*100)}"{bold} dirty="0">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:latin typeface="{FONT}" pitchFamily="34" charset="0"/>'
            f'<a:ea typeface="{FONT}" pitchFamily="34" charset="-122"/>'
            f'<a:cs typeface="{FONT}" pitchFamily="34" charset="-120"/></a:rPr>')

def run(text, sz, b=False, color=DARK):
    return f'<a:r>{rpr(sz, b, color)}<a:t>{escape(text)}</a:t></a:r>'

def para(runs, sz=12, b=False, color=DARK, bullet=False, align="l", spc_after=0, spc_before=0, lnspc=None, bu_color=MAG, marL=None):
    if isinstance(runs, str):
        runs = [(runs, {})]
    ppr = f'<a:pPr algn="{align}"'
    if bullet:
        ppr += ' marL="228600" indent="-228600"'
    elif marL is not None:
        ppr += f' marL="{E(marL)}" indent="0"'
    else:
        ppr += ' marL="0" indent="0"'
    ppr += '>'
    if lnspc:
        ppr += f'<a:lnSpc><a:spcPct val="{int(lnspc*1000)}"/></a:lnSpc>'
    if spc_before:
        ppr += f'<a:spcBef><a:spcPts val="{int(spc_before*100)}"/></a:spcBef>'
    if spc_after:
        ppr += f'<a:spcAft><a:spcPts val="{int(spc_after*100)}"/></a:spcAft>'
    if bullet:
        ppr += f'<a:buClr><a:srgbClr val="{bu_color}"/></a:buClr><a:buSzPct val="100000"/><a:buFont typeface="Arial"/><a:buChar char="&#x2022;"/>'
    else:
        ppr += '<a:buNone/>'
    ppr += '</a:pPr>'
    body = "".join(run(t, o.get("sz", sz), o.get("b", b), o.get("color", color)) for t, o in runs)
    return f'<a:p>{ppr}{body}</a:p>'

def textbox(x, y, w, h, paras, anchor="t", ins=0.0, name="Text"):
    i = E(ins)
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name="{name}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{E(x)}" y="{E(y)}"/><a:ext cx="{E(w)}" cy="{E(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="{i}" tIns="{i}" rIns="{i}" bIns="{i}" rtlCol="0" anchor="{anchor}"/>'
            f'<a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>')

def shape(x, y, w, h, fill, prst="roundRect", adj=12000, line=None, name="Shape"):
    geom = f'<a:prstGeom prst="{prst}"><a:avLst>' + (f'<a:gd name="adj" fmla="val {adj}"/>' if prst == "roundRect" else "") + '</a:avLst></a:prstGeom>'
    ln = f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>' if line else '<a:ln><a:noFill/></a:ln>'
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name="{name}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{E(x)}" y="{E(y)}"/><a:ext cx="{E(w)}" cy="{E(h)}"/></a:xfrm>{geom}'
            f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{ln}</p:spPr>'
            f'<p:txBody><a:bodyPr rtlCol="0" anchor="ctr"/><a:lstStyle/><a:p><a:pPr algn="ctr"/><a:endParaRPr lang="ja-JP" sz="1000"/></a:p></p:txBody></p:sp>')

def badge(x, y, d, text, fill):
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name="Badge"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{E(x)}" y="{E(y)}"/><a:ext cx="{E(d)}" cy="{E(d)}"/></a:xfrm>'
            f'<a:prstGeom prst="ellipse"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr>'
            f'<p:txBody><a:bodyPr lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" anchor="ctr"/><a:lstStyle/>'
            f'{para(text, sz=14, b=True, color=WHITE, align="ctr")}</p:txBody></p:sp>')

def cell(text, sz=10, b=False, color=DARK, fill=None, align="l"):
    if isinstance(text, str):
        text = [(text, {})]
    p = para(text, sz=sz, b=b, color=color, align=align)
    ln = lambda tag: f'<a:{tag} w="6350"><a:solidFill><a:srgbClr val="{LINE}"/></a:solidFill></a:{tag}>'
    f = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else '<a:noFill/>'
    return (f'<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>{p}</a:txBody>'
            f'<a:tcPr marL="54864" marR="54864" marT="27432" marB="27432" anchor="ctr">{ln("lnL")}{ln("lnR")}{ln("lnT")}{ln("lnB")}{f}</a:tcPr></a:tc>')

def table(x, y, colw, rows, rowh=0.3, sz=10, header_fill=PUR, name="Table", header_h=None, header_sz=None):
    grid = "".join(f'<a:gridCol w="{E(w)}"/>' for w in colw)
    header_h = rowh if header_h is None else header_h
    header_sz = sz if header_sz is None else header_sz
    trs = []
    for ri, row in enumerate(rows):
        tcs = []
        for ci, item in enumerate(row):
            t, o = (item if isinstance(item, tuple) else (item, {}))
            if ri == 0:
                tcs.append(cell(t, sz=header_sz, b=True, color=WHITE, fill=header_fill, align=o.get("align", "l" if ci == 0 else "ctr")))
            else:
                tcs.append(cell(t, sz=o.get("sz", sz), b=o.get("b", False), color=o.get("color", DARK),
                                fill=o.get("fill"), align=o.get("align", "l" if ci == 0 else "ctr")))
        trs.append(f'<a:tr h="{E(header_h if ri == 0 else rowh)}">{"".join(tcs)}</a:tr>')
    h = header_h + rowh * (len(rows) - 1)
    return (f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{nid()}" name="{name}"/>'
            f'<p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>'
            f'<p:xfrm><a:off x="{E(x)}" y="{E(y)}"/><a:ext cx="{E(sum(colw))}" cy="{E(h)}"/></p:xfrm>'
            f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
            f'<a:tbl><a:tblPr firstRow="1" bandRow="1"/><a:tblGrid>{grid}</a:tblGrid>{"".join(trs)}</a:tbl>'
            f'</a:graphicData></a:graphic></p:graphicFrame>')

_rid = [90]
def picture(n, png, x, y, w, h):
    """Copy png into ppt/media, add an image relationship to slide n, return a <p:pic>."""
    _rid[0] += 1
    rid = f"rId{_rid[0]}"
    media = os.path.join(ROOT, "ppt", "media")
    name = os.path.basename(png)
    shutil.copy(png, os.path.join(media, name))
    rp = os.path.join(SL, "_rels", f"slide{n}.xml.rels")
    s = open(rp, encoding="utf-8").read()
    s = s.replace("</Relationships>", f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{name}"/></Relationships>')
    open(rp, "w", encoding="utf-8").write(s)
    return (f'<p:pic><p:nvPicPr><p:cNvPr id="{nid()}" name="{name}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr><a:xfrm><a:off x="{E(x)}" y="{E(y)}"/><a:ext cx="{E(w)}" cy="{E(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')

# ---------- layout constants (inches) ----------
X0 = 0.234; W = 8.70; Y0 = 0.99; XR = X0 + W

def write_slide(n, title, shapes, title_old="今週までのまとめ", drop_text1=False):
    path = os.path.join(SL, f"slide{n}.xml")
    s = open(path, encoding="utf-8").read()
    assert f"<a:t>{title_old}</a:t>" in s, (n, title_old)
    s = s.replace(f"<a:t>{title_old}</a:t>", f"<a:t>{escape(title)}</a:t>", 1)
    if drop_text1:
        s = re.sub(r'<p:sp><p:nvSpPr><p:cNvPr id="3" name="Text 1">.*?</p:sp>', "", s, count=1, flags=re.S)
    s = s.replace("</p:spTree>", "".join(shapes) + "</p:spTree>", 1)
    open(path, "w", encoding="utf-8").write(s)

def foot(text, w=W, x=X0):
    return textbox(x, 4.93, w, 0.3, [para(text, sz=9, color=GRAY)])

# ---------- slide 1: cover ----------
write_slide(1, "2026年第44週　進捗報告", [], title_old="2025年第 週　進捗報告")

# ---------- slide 2: 今週までのまとめ ----------
items = [
    ("V7（スペクトル走査 Mamba）の学習完了と評価", MAG,
     "10/21 のマシン停止で 1 週間中断、10/28 再起動後に 227 epoch から再開して 300 epoch 完了。テスト PSNR 38.86 dB：V 系列では最良（Base +0.11、V6 +0.22）だが、直接の対照 HyMCNetL3（空間走査、39.12）には 0.26 dB 届かない。"),
    ("帯域診断：低域は学べている、負けているのは中域", MID,
     "帯域別 SNR とサブバンド入替（oracle）で確認。V7 の低域は SSR-S・Base より良いが、16〜128 スケールの中低域で空間走査に劣る。8×8 ウィンドウ内走査が中スケールの空間文脈を失うことが原因。"),
    ("結論：主線を「ウェーブレットサンプリング＋空間走査 Mamba」へ", PUR,
     "V6（ブロック内帯域分割）と V7（スペクトル走査）はいずれも明確な負の結論。現時点で有効と確認できた Mamba の使い方は HyMCNetL3 の形のみ。スペクトル走査は Chikusei 128 バンドで最終検証する。"),
]
sh = []
y = Y0; ch = 1.15; gap = 0.14
for i, (t, col, d) in enumerate(items):
    sh.append(shape(X0, y, W, ch, CARD, line=LINE))
    sh.append(badge(X0 + 0.22, y + (ch - 0.5) / 2, 0.5, str(i + 1), col))
    sh.append(textbox(X0 + 0.95, y + 0.14, W - 1.2, 0.32, [para(t, sz=14, b=True)]))
    sh.append(textbox(X0 + 0.95, y + 0.48, W - 1.2, ch - 0.55, [para(d, sz=10.5, color=BODY, lnspc=110)]))
    y += ch + gap
sh.append(foot("タスク：CASSI 28 バンド再構成、3-stage 展開ネットワーク ｜ CAVE で学習、10 シーンで評価 ｜ 前回（10/19）以降の進捗"))
write_slide(2, "今週までのまとめ", sh)

# ---------- slide 4: 主結果 ----------
T = {"fill": TINT}; TB = {"b": True, "fill": TINT}
hdr = ["モデル", "構成の要点", "パラメータ", "PSNR (dB)", "SSIM", "SAM (°)", "学習 PSNR"]
rows = [hdr,
    ["SSR-S（原版 3 stage）", "SSRB＋ストライド畳み込み", "1.81M", "38.93", "0.9731", "4.20", "41.04"],
    ["SSR-M（原版 6 stage）", "同上", "3.62M", "39.94", "0.9777", "3.67", "42.86"],
    ["SSR-L（原版 9 stage）", "同上", "5.43M", "40.47", "0.9799", "3.33", "44.06"],
    [("HyMCNetL3", TB), ("MamConvB＋WT up/down（空間走査）", T), ("2.94M", T), ("39.12", TB), ("0.9742", T), ("3.78", T), ("43.39", T)],
    [("V7（今週）", TB), ("HyMCNetL3 の Mamba をスペクトル走査に", T), ("2.95M", T), ("38.86", TB), ("0.9728", T), ("3.96", T), ("43.20", T)],
    ["Base", "MamConvB＋SSR ストライド畳み込み", "2.04M", "38.75", "0.9724", "4.00", "42.70"],
    ["V6", "ブロック内 WT 二分岐＋Hilbert＋構造トークン＋2D 畳み込み", "3.15M", "38.64", "0.9712", "4.21", "42.57"],
    ["V5", "ブロック内 WT 二分岐＋SSR 初期化", "2.90M", "38.51", "0.9706", "4.15", "42.47"],
    ["V4", "ブロック内 WT 二分岐", "2.04M", "38.21", "0.9690", "4.40", "42.11"],
]
rh = 0.255
sh = [table(X0, Y0, [1.6, 3.15, 0.8, 0.8, 0.8, 0.7, 0.85], rows, rowh=rh, sz=9)]
bul = [
    [("V7 は V 系列で最良", {"b": True}), ("（Base +0.11、V6 +0.22、SSIM・SAM も同様）。ただし HyMCNetL3 より 0.26 dB、SSR-S より 0.07 dB 低い", {})],
    [("291〜300 epoch のテスト PSNR は 38.840〜38.858 で収束済み。学習-テスト差は 4.3 dB で HyMCNetL3（4.27）と同等、汎化は改善せず", {})],
    [("シーン別（対 Base）：scene09 +0.66、scene03 +0.47、scene02 +0.29 ／ scene05 −0.21、scene01 −0.19。", {}), ("全面的な改善ではない", {"b": True})],
]
sh.append(textbox(X0, Y0 + rh * 10 + 0.12, W, 1.15, [para(b, sz=10, bullet=True, spc_after=3, lnspc=108) for b in bul]))
sh.append(foot("同一テストセット 10 シーン、PSNR はシーン平均、291〜300 epoch の最良 ｜ 推論 256×256 1 シーン：V7 26.4 ms、V6 26.6 ms、Base 21.0 ms（同一 GPU）"))
write_slide(4, "主結果：V7 は V 系列最良だが HyMCNetL3 に届かず", sh)

# ---------- slide 5: V6 振り返り ----------
sh = [textbox(X0, Y0, W, 0.5, [para([("仮説：", {"b": True}), ("V5 のブロック内低域 Mamba は行優先の 1D 因果走査で上下隣接画素が系列上 128 離れ、2D 構造が見えない。走査方向を増やさず", {}), ("符号化だけで補う", {"b": True}), ("（他は V5 と同一、3.15M、初期偏差 0.072 ≈ 恒等）。", {})], sz=10.5, lnspc=110)])]
cards = [
    ("1　構造トークン", MAG, "4 サブバンドを 1 トークンに", "LL / HL / LH / HH を連結し 1×1 で C 次元へ。各トークンが 2×2 近傍の「平均＋水平＋垂直＋対角」を持つ。出力は LL のみに加算。"),
    ("2　Hilbert 曲線走査", MID, "隣接トークンの 2D 距離を常に 1 に", "行優先では平均約 126。パラメータ 0、固定置換 1 回。"),
    ("3　2D 深さ方向畳み込み", PUR, "Mamba 内の 1D 因果畳み込みを置換", "走査前にトークンが 2D 近傍を保持。状態空間パラメータは mamba_ssm の初期化のまま。"),
]
cy = 1.5; cw = (W - 0.3) / 3; chh = 1.45
for i, (lab, col, t, d) in enumerate(cards):
    cx = X0 + i * (cw + 0.15)
    sh.append(shape(cx, cy, cw, chh, CARD, line=LINE))
    sh.append(textbox(cx + 0.18, cy + 0.12, cw - 0.36, 0.26, [para(lab, sz=10, b=True, color=col)]))
    sh.append(textbox(cx + 0.18, cy + 0.38, cw - 0.36, 0.3, [para(t, sz=11, b=True, lnspc=105)]))
    sh.append(textbox(cx + 0.18, cy + 0.7, cw - 0.36, chh - 0.78, [para(d, sz=9, color=BODY, lnspc=108)]))
ty = cy + chh + 0.18
LW = 4.3; RX = X0 + LW + 0.25; RW = W - LW - 0.25
sh.append(textbox(X0, ty, LW, 0.26, [para("結果（10/17 学習完了）と サブバンド入替", sz=11, b=True, color=MID)]))
rows = [["比較（低域スケール）", "V6 低域＋相手 高域", "相手 低域＋V6 高域"],
        ["vs HyMCNetL3（128×128）", "38.72", ("39.03", {"b": True, "fill": TINT})],
        ["vs HyMCNetL3（16×16）", "39.00", "38.75"],
        ["vs V5（16×16）", "38.49", ("38.65", {"b": True, "fill": TINT})]]
sh.append(table(X0, ty + 0.3, [1.9, 1.2, 1.2], rows, rowh=0.25, sz=9))
sh.append(textbox(X0, ty + 0.3 + 0.25 * 4 + 0.04, LW, 0.3, [para("単体 PSNR：V6 38.64 ／ V5 38.51 ／ Base 38.75 ／ HyMCNetL3 39.12", sz=8.5, color=GRAY)]))
sh.append(shape(RX, ty, RW, 1.7, PUR))
sh.append(textbox(RX + 0.18, ty + 0.12, RW - 0.36, 0.3, [para("結論：ブロック内帯域分割は打ち切り", sz=11.5, b=True, color=WHITE)]))
rp = [
    "V6 38.64：V5 +0.13 だが、分割しない Base に −0.11、HyMCNetL3 に −0.48",
    "V6 の 128×128 低域を HyMCNetL3 のものに替えると +0.39、高域だけ替えても +0.08 → 弱点は依然として低域",
    "16×16 で「V5 低域＋V6 高域」(38.65) ≥ V6 自身 (38.64)：符号化は低域 Mamba に構造を学ばせられず、増分は細部サブバンド由来",
]
sh.append(textbox(RX + 0.18, ty + 0.45, RW - 0.36, 1.25, [para(r, sz=9, color="F3EAF8", bullet=True, spc_after=3, lnspc=106, bu_color="F3A6CC") for r in rp]))
write_slide(5, "V6 振り返り：符号化で 1D Mamba に 2D 構造を与える試み", sh)

# ---------- slide 6: V7 設計 ----------
sh = [textbox(X0, Y0, W, 0.5, [para([("動機：", {"b": True}), ("CASSI の結合はスペクトル方向にある。空間 1D 走査は 2D 画像上で構造的に制約されるが、各画素のバンド列は自然に順序付いている。", {}), ("Mamba に最も適した走査軸はどちらか", {"b": True}), ("を、他を一切変えずに検証する。", {})], sz=10.5, lnspc=110)])]
cards = [
    ("変更は 1 か所のみ", MAG, "Mamba_Att → SpecMamba_Att",
     "HyMCNetL3 の各 MambaB で、H×W ラスタ走査の Mamba を置換。特徴マップを 8×8 ウィンドウに分割し、各ウィンドウを「チャネルを系列、64 画素をトークン特徴」とする系列にして、Mamba を波長順に走査する（SSR の WSSA と同じトークン配置、注意の代わりに選択的走査）。"),
    ("空間構造は畳み込みに任せる", MID, "5×5 位置畳み込み＋11×11 ConvB＋U-Net",
     "ウィンドウ間は走査で混ざらない。ウィンドウをまたぐ情報は畳み込みと Haar 多重スケール経路だけが運ぶ。骨格・WT サンプリング・損失・学習設定は HyMCNetL3 と同一なので、HyMCNetL3 vs V7 の差分は走査軸のみ。"),
    ("実装検証（10/19）", PUR, "因果性・独立性・コスト",
     "第 40 チャネル以降を摂動しても前 40 チャネルの出力は不変（スペクトル因果性）。ウィンドウ間は走査上独立。NaN なし。パラメータ・速度・VRAM は HyMCNetL3 と同等。"),
]
cy = 1.5; cw = (W - 0.3) / 3; chh = 2.3
for i, (lab, col, t, d) in enumerate(cards):
    cx = X0 + i * (cw + 0.15)
    sh.append(shape(cx, cy, cw, chh, CARD, line=LINE))
    sh.append(textbox(cx + 0.2, cy + 0.15, cw - 0.4, 0.28, [para(lab, sz=10.5, b=True, color=col)]))
    sh.append(textbox(cx + 0.2, cy + 0.45, cw - 0.4, 0.56, [para(t, sz=12, b=True, lnspc=110)]))
    sh.append(textbox(cx + 0.2, cy + 1.0, cw - 0.4, chh - 1.1, [para(d, sz=9, color=BODY, lnspc=108)]))
tiles = [("パラメータ数（HyMCNetL3 2.94M）", "2.95M"), ("1 step / VRAM", "0.27 s / 10.6 GB"), ("学習期間", "10/19 → 10/28（停止 1 週）"), ("学習 PSNR（300 ep）", "43.20 dB")]
ty = 3.95; tw = (W - 0.45) / 4; th = 0.85
for i, (lab, val) in enumerate(tiles):
    tx = X0 + i * (tw + 0.15)
    sh.append(shape(tx, ty, tw, th, TINT))
    sh.append(textbox(tx + 0.15, ty + 0.1, tw - 0.3, 0.25, [para(lab, sz=8.5, color=GRAY)]))
    sh.append(textbox(tx + 0.15, ty + 0.38, tw - 0.3, 0.4, [para(val, sz=12.5, b=True)]))
sh.append(foot("リスク（設計時）：深い層のチャネル（56 / 112）は物理バンドに対応せず「スペクトル順」は第 1 層でのみ厳密 ／ ウィンドウ間の情報は畳み込み頼み"))
write_slide(6, "V7：Mamba の走査軸を空間からスペクトルへ", sh)

# ---------- slide 7: V7 帯域診断 ① ----------
LW = 4.3; RX = X0 + LW + 0.3; RW = W - LW - 0.3
sh = [textbox(X0, Y0, LW, 0.28, [para("① 帯域別 SNR − SSR-S（dB、Haar 2 段、各モデル epoch 300）", sz=11, b=True, color=MID)])]
sh.append(picture(7, os.path.join(HERE, "fig_band_snr.png"), X0, Y0 + 0.32, LW, LW / (4.3 / 2.75)))
ty = Y0 + 0.32 + LW / (4.3 / 2.75) + 0.08
rows = [["対 SSR-S", "LL2", "HL2", "LH2", "HH2", "HL1", "LH1", "HH1"],
        ["HyMCNetL3", ("+0.24", {"b": True}), "−0.09", "−0.14", "−0.13", "−0.17", "−0.12", "−0.10"],
        [("V7", {"b": True}), "−0.09", "−0.26", ("−0.57", {"b": True}), "−0.33", "−0.33", "−0.41", "−0.24"],
        ["Base", "−0.21", "−0.25", "−0.37", "−0.31", "−0.30", "−0.33", "−0.28"]]
sh.append(table(X0, ty, [0.94] + [0.48] * 7, rows, rowh=0.23, sz=8))
sh.append(textbox(RX, Y0, RW, 0.28, [para("読み取り", sz=11, b=True, color=MID)]))
bul = [
    [("最低域 LL2：", {"b": True}), ("V7 25.57 dB は Base (25.45) より上、SSR-S (25.66)・HyMCNetL3 (25.90) より下。", {}), ("低域は学べているが空間走査には及ばない", {"b": True})],
    [("細部サブバンド：", {"b": True}), ("V7 は全帯域で HyMCNetL3 に劣り、", {}), ("LH2（64×64 スケールの水平エッジ）が最悪", {"b": True}), ("（−0.43 dB）。最高域 HH1 は Base より僅かに良い", {})],
    [("誤差の分布：", {"b": True}), ("V7 も他モデルと同様に低域主体（LL2 が誤差エネルギーの 41%、FFT 低域リング 75%）。PSNR を決めるのは依然として低域構造", {})],
    [("HyMCNetL3 だけが LL2 で SSR-S に勝つ", {"b": True}), ("という前回の所見は変わらず。V7 の LL2 は HyMCNetL3 に 0.33 dB 届かない", {})],
]
sh.append(textbox(RX, Y0 + 0.36, RW, 3.6, [para(b, sz=10, bullet=True, spc_after=6, lnspc=110) for b in bul]))
sh.append(foot("analyze_freq.py → Result_freq3/freq_analysis.csv ｜ 帯域 SNR = その帯域の信号エネルギー ／ 誤差エネルギー"))
write_slide(7, "帯域診断 ①：V7 の低域は Base より良く、空間走査より悪い", sh)

# ---------- slide 8: V7 帯域診断 ②（サブバンド入替） ----------
LW = 5.3; RX = X0 + LW + 0.25; RW = W - LW - 0.25
sh = [textbox(X0, Y0, LW, 0.28, [para("② サブバンド入替（oracle band-swap）：低域を V7 から、細部を相手から", sz=11, b=True, color=MID)])]
sh.append(picture(8, os.path.join(HERE, "fig_swap.png"), X0, Y0 + 0.32, LW, LW / (5.4 / 2.75)))
fy = Y0 + 0.32 + LW / (5.4 / 2.75)
rows = [["低域スケール", "V7 低域＋HyM 高域", "HyM 低域＋V7 高域", "V7 低域＋SSR-S 高域", "SSR-S 低域＋V7 高域"],
        ["128×128", "38.90", ("39.07", {"b": True}), "38.90", "38.87"],
        ["64×64", "38.99", "38.97", ("39.00", {"b": True, "fill": TINT}), "38.78"],
        ["16×16", ("39.08", {"b": True}), "38.88", ("39.01", {"b": True, "fill": TINT}), "38.77"]]
sh.append(table(X0, fy + 0.06, [0.95, 1.09, 1.09, 1.09, 1.08], rows, rowh=0.22, sz=8, header_h=0.3, header_sz=7.5))
sh.append(textbox(RX, Y0, RW, 0.28, [para("読み取り", sz=11, b=True, color=MID)]))
bul = [
    [("対 SSR-S：", {"b": True}), ("「V7 低域＋SSR-S 高域」は 64×64 以下で 39.00 と", {}), ("両者を同時に上回り", {"b": True}), ("、逆は同時に下回る。V7 の低域は確かに SSR-S より良く、足を引くのは高域細部。HyMCNetL3 と同じ型（ただし HyMCNetL3 は 39.13）", {})],
    [("対 HyMCNetL3：", {"b": True}), ("V7 の 16×16 低域だけ残して他を HyMCNetL3 にすると 39.08（差 0.04）、V7 の最細部だけ残しても −0.05。", {}), ("差は 16〜128 の中間スケールに分散", {"b": True}), ("し、1 段上がるごとに 0.04〜0.09 dB ずつ失う", {})],
]
sh.append(textbox(RX, Y0 + 0.36, RW, 2.45, [para(b, sz=9.5, bullet=True, spc_after=5, lnspc=108) for b in bul]))
by = 3.55; bh = 1.6
sh.append(shape(RX, by, RW, bh, MAG))
sh.append(textbox(RX + 0.16, by + 0.12, RW - 0.32, 0.5, [para("結論：壊れたのはスペクトル側ではなく、ウィンドウ内走査が失った中スケールの空間文脈", sz=10.5, b=True, color=WHITE, lnspc=108)]))
sh.append(textbox(RX + 0.16, by + 0.7, RW - 0.32, 0.9, [para("空間走査の HyMCNetL3 は画像全体を見るが、V7 のウィンドウ間情報は 5×5 / 11×11 畳み込みと Haar 経路だけ。救うなら、ウィンドウ拡大か空間＋スペクトル並列走査。", sz=9, color="FBE3EF", lnspc=108)]))
write_slide(8, "帯域診断 ②：負けているのは 16〜128 の中低域", sh)

# ---------- slide 9: 結論と位置づけ ----------
rows = [["比較", "唯一の変更点（骨格・学習設定は同一）", "PSNR 変化"],
    ["SSR-S → Base", "SSRB ブロックを MamConvB ブロックに置換", "−0.18 dB"],
    [("Base → HyMCNetL3", TB), ("ストライド畳み込みをウェーブレット up/down に置換", T), ("+0.37 dB", TB)],
    ["Base → V5", "ブロック内 WT 二分岐（低域 Mamba ／ 高域畳み込み）", "−0.24 dB"],
    ["Base → V6", "ブロック内 WT 二分岐＋3 つの符号化改善", "−0.11 dB"],
    [("HyMCNetL3 → V7", TB), ("Mamba の走査軸：空間（H×W）→ スペクトル（8×8 窓内のチャネル）", T), ("−0.26 dB", TB)],
]
sh = [table(X0, Y0, [1.9, 5.3, 1.5], rows, rowh=0.29, sz=10)]
sh.append(textbox(X0, Y0 + 0.29 * 6 + 0.08, W, 0.3, [para("正の寄与はサンプリング層のウェーブレットのみ。「より精密な」Mamba の使い方（ブロック内分割、スペクトル走査）はいずれも WT 骨格上の空間走査に劣る。", sz=10, color=GRAY, lnspc=106)]))
cards = [
    ("主線は HyMCNetL3 の形", "ウェーブレットサンプリング＋空間走査 Mamba を本体に据え、論文の主張をここに絞る"),
    ("V6・V7 はアブレーションとして残す", "「ブロック内帯域分割」「走査軸」の 2 つの負の結果が、設計選択の根拠になる"),
    ("スペクトル走査は 128 バンドで再検証", "28 バンドでは系列が短すぎる可能性。Chikusei（128 バンド）の V7NB vs NB が最終判定"),
]
cy = 3.1; cw = (W - 0.3) / 3; chh = 1.65
for i, (t, d) in enumerate(cards):
    cx = X0 + i * (cw + 0.15)
    sh.append(shape(cx, cy, cw, chh, PUR))
    sh.append(textbox(cx + 0.2, cy + 0.17, cw - 0.4, 0.28, [para(f"結論 {i+1}", sz=10, b=True, color="F3A6CC")]))
    sh.append(textbox(cx + 0.2, cy + 0.47, cw - 0.4, 0.6, [para(t, sz=12.5, b=True, color=WHITE, lnspc=110)]))
    sh.append(textbox(cx + 0.2, cy + 1.05, cw - 0.4, 0.55, [para(d, sz=9.5, color="E6D8F2", lnspc=108)]))
sh.append(foot("PSNR 変化は 10 シーン平均の差。テストセットが小さいため 0.1〜0.2 dB の差は慎重に解釈する"))
write_slide(9, "アブレーション全体像と研究の位置づけ", sh)

# ---------- slide 3: 予定 ----------
LW = 5.3; RX = X0 + LW + 0.3; RW = W - LW - 0.3
plans = [
    ("主線（HyMCNetL3）の補強", "FFN 二重残差を修正した HyMCNetL3 を 1 本学習して影響を定量化（安価な利得の可能性）。4 / 6 stage 版を SSR-M と同パラメータで比較"),
    ("Chikusei 128 バンド線の開始", "run_chik.sh で V7NB → NB 対照の順に学習（128×128 パッチ、mult 1,1,2）。スペクトル走査が長い系列で逆転するかを判定。有効なら V7S 三方式（σ = 4 → 2 / 8）へ"),
    ("V7 の救済案（優先度低）", "ウィンドウ 8 → 16、または空間＋スペクトル並列走査。中スケールの空間文脈不足への対処"),
    ("論文用指標の再確認", "SSIM を MATLAB の cal_psnr_ssim.m で再計算し、論文と同じ基準に揃える"),
]
paras = []
for t, d in plans:
    paras.append(para(t, sz=11, b=True, bullet=True, spc_before=3))
    paras.append(para(d, sz=10, color=BODY, spc_after=5, lnspc=108, marL=0.25))
sh = [textbox(X0, Y0, LW, 3.9, paras)]
sh.append(foot("優先順位：1 と 2 を並行（GPU 1 枚、各 300 epoch ≈ 2 日）、3 は 2 の結果次第", w=LW))
sh.append(shape(RX, Y0, RW, 3.3, PUR))
sh.append(textbox(RX + 0.2, Y0 + 0.15, RW - 0.4, 0.3, [para("リスク", sz=12, b=True, color="F3A6CC")]))
risks = [
    "マシンが 10/21 に停止し 1 週間を失った。自動再起動と watchdog は設定済みだが BIOS 側の設定は未完",
    "Chikusei は学習タイルが 35 枚と少なく、CAVE と同じ結論が出るとは限らない",
    "テストセットは 10 シーンと小さく、0.1〜0.2 dB の差は慎重に解釈する",
]
sh.append(textbox(RX + 0.2, Y0 + 0.5, RW - 0.4, 2.7, [para(r, sz=10, color="F3EAF8", bullet=True, spc_after=5, lnspc=110, bu_color="F3A6CC") for r in risks]))
write_slide(3, "予定とリスク", sh, title_old="予定", drop_text1=True)

# ---------- slide order: 1, 2, 4..9, 3 ----------
pp = os.path.join(ROOT, "ppt", "presentation.xml")
s = open(pp, encoding="utf-8").read()
order = ['<p:sldId id="256" r:id="rId2"/>', '<p:sldId id="257" r:id="rId3"/>'] + \
        [f'<p:sldId id="{259 + i}" r:id="rId{11 + i}"/>' for i in range(6)] + ['<p:sldId id="258" r:id="rId4"/>']
s = re.sub(r'<p:sldIdLst>.*?</p:sldIdLst>', '<p:sldIdLst>' + "".join(order) + '</p:sldIdLst>', s, flags=re.S)
open(pp, "w", encoding="utf-8").write(s)
print("done")
