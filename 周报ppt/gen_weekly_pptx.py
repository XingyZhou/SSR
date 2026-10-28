#!/usr/bin/env python3
"""Fill the Rikkyo AI weekly-report template (unpacked pptx) with this week's content."""
import re, sys, os
from xml.sax.saxutils import escape

ROOT = sys.argv[1]  # unpacked dir
SL = os.path.join(ROOT, "ppt", "slides")
EMU = 914400
FONT = "IBM Plex Sans JP"
DARK = "1D2129"; GRAY = "6B6480"; MAG = "D6307F"; PUR = "4B2A8C"; MID = "8A2F8C"
TINT = "F6E6F0"; CARD = "F7F4FA"; LINE = "E3DCEA"; WHITE = "FFFFFF"

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

def para(runs, sz=12, b=False, color=DARK, bullet=False, align="l", spc_after=0, spc_before=0, lnspc=None):
    """runs: str or list of (text, dict) tuples."""
    if isinstance(runs, str):
        runs = [(runs, {})]
    ppr = f'<a:pPr algn="{align}"'
    if bullet:
        ppr += ' marL="228600" indent="-228600"'
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
        ppr += f'<a:buClr><a:srgbClr val="{MAG}"/></a:buClr><a:buSzPct val="100000"/><a:buFont typeface="Arial"/><a:buChar char="&#x2022;"/>'
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
    """Circle with a white number."""
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
    """rows: list of lists; each item str or (str, opts). First row is header.
    header_h / header_sz override the header row's height and font size (for wrapped headers)."""
    grid = "".join(f'<a:gridCol w="{E(w)}"/>' for w in colw)
    header_h = rowh if header_h is None else header_h
    header_sz = sz if header_sz is None else header_sz
    trs = []
    for ri, row in enumerate(rows):
        tcs = []
        for ci, item in enumerate(row):
            if isinstance(item, tuple):
                t, o = item
            else:
                t, o = item, {}
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

# ---------- slide 1: cover ----------
write_slide(1, "2026年第42週　進捗報告", [], title_old="2025年第 週　進捗報告")

# ---------- slide 2: 今週までのまとめ ----------
items = [
    ("学習と同条件比較", MAG,
     "V4・V5・ベースライン Base の 300 epoch 学習を完了。同一テストセット（CAVE 10 シーン）・同一評価コードで SSR 原版（S / M / L）と HyMCNetL3 と比較した。"),
    ("診断ツールを 2 つ構築", MID,
     "「帯域分割の再構成は低域と高域のどちらで負けているか」に答えるため、帯域別誤差分解（Haar 2 段サブバンド＋FFT 径方向リング）とサブバンド入替（oracle band-swap）実験を実装した。"),
    ("V6 の設計と学習開始", PUR,
     "診断結果に基づき、走査方向を増やさずに符号化で 1D Mamba に 2D 構造を見せる V6 を設計。本日（10/15）300 epoch の学習を開始、約 2 日で完了予定。"),
]
sh = []
y = Y0; ch = 1.15; gap = 0.14
for i, (t, col, d) in enumerate(items):
    sh.append(shape(X0, y, W, ch, CARD, line=LINE))
    sh.append(badge(X0 + 0.22, y + (ch - 0.5) / 2, 0.5, str(i + 1), col))
    sh.append(textbox(X0 + 0.95, y + 0.14, W - 1.2, 0.32, [para(t, sz=14, b=True)]))
    sh.append(textbox(X0 + 0.95, y + 0.48, W - 1.2, ch - 0.55, [para(d, sz=11, color="3A3550", lnspc=110)]))
    y += ch + gap
sh.append(textbox(X0, 4.93, W, 0.3, [para("タスク：CASSI 28 バンド再構成、3-stage 展開ネットワーク ｜ CAVE で学習、10 シーンで評価", sz=9.5, color=GRAY)]))
write_slide(2, "今週までのまとめ", sh)

# ---------- slide 4: 主結果 ----------
hdr = ["モデル", "構成の要点", "パラメータ", "PSNR (dB)", "SSIM", "SAM (°)", "学習 PSNR"]
rows = [hdr,
    ["SSR-S（原版 3 stage）", "SSRB＋ストライド畳み込み", "1.81M", "38.93", "0.9731", "4.20", "41.04"],
    ["SSR-M（原版 6 stage）", "同上", "3.62M", "39.94", "0.9777", "3.67", "42.86"],
    ["SSR-L（原版 9 stage）", "同上", "5.43M", "40.47", "0.9799", "3.33", "44.06"],
    [("HyMCNetL3", {"b": True, "fill": TINT}), ("MamConvB＋WT up/down", {"fill": TINT}), ("2.94M", {"fill": TINT}), ("39.12", {"b": True, "fill": TINT}), ("0.9742", {"fill": TINT}), ("3.78", {"fill": TINT}), ("43.39", {"fill": TINT})],
    ["Base（今週）", "MamConvB＋SSR ストライド畳み込み", "2.04M", "38.75", "0.9724", "4.00", "42.70"],
    ["V5（今週）", "ブロック内 WT 二分岐＋SSR 初期化", "2.90M", "38.51", "0.9706", "4.15", "42.47"],
    ["V4（今週）", "ブロック内 WT 二分岐", "2.04M", "38.21", "0.9690", "4.40", "42.11"],
    ["V3（先週）", "ブロック内分解、フル解像度ブロックなし", "1.25M", "36.55", "0.9570", "5.64", "39.49"],
]
sh = [table(X0, Y0, [1.75, 2.95, 0.8, 0.8, 0.8, 0.75, 0.85], rows, rowh=0.29, sz=9.5)]
bul = [
    [("V3 → V4 で ", {}), ("+1.66 dB", {"b": True}), ("（フル解像度ブロックと残差接続を復元）、V4 → V5 で ", {}), ("+0.29 dB", {"b": True}), ("（高域 ConvB 非共有・SSR 式初期化・FFN 二重残差の修正）", {})],
    [("V5 は SSR-S より 0.42 dB、HyMCNetL3 より 0.61 dB 低い。パラメータ数は HyMCNetL3 と同等なので", {}), ("容量不足ではない", {"b": True})],
    [("Mamba 系は学習 PSNR が SSR-S より 1.4〜2.4 dB 高いのにテストで勝てない：汎化差は構造由来。推論時間（相対）は SSR-S 27.8 ms、HyMCNetL3 56.5 ms、Base 53.2 ms、V5 49.4 ms で、ブロック内 WT による高速化は見られない", {})],
]
sh.append(textbox(X0, Y0 + 0.29 * 9 + 0.15, W, 1.25, [para(b, sz=10.5, bullet=True, spc_after=3, lnspc=108) for b in bul]))
sh.append(textbox(X0, 4.98, W, 0.25, [para("同一テストセット 10 シーン、PSNR はシーン平均 ｜ 推論時間は 256×256 の 1 シーン、同一 GPU", sz=9, color=GRAY)]))
write_slide(4, "主結果：V5 は改善も SSR-S・HyMCNetL3 に届かず", sh)

# ---------- slide 5: アブレーション ----------
rows = [["比較", "変更点（骨格・学習設定は同一）", "PSNR 変化"],
    ["SSR-S → Base", "SSRB ブロックを MamConvB ブロックに置換", "−0.18 dB"],
    [("Base → HyMCNetL3", {"fill": TINT}), ("ストライド畳み込みをウェーブレット up/down に置換", {"fill": TINT}), ("+0.37 dB", {"b": True, "fill": TINT})],
    [("Base → V5", {"fill": TINT}), ("MamConvB ブロックをブロック内 WT 二分岐に置換", {"fill": TINT}), ("−0.24 dB", {"b": True, "fill": TINT})],
]
sh = [table(X0, Y0, [2.0, 5.2, 1.5], rows, rowh=0.32, sz=10.5)]
sh.append(textbox(X0, Y0 + 0.32 * 4 + 0.1, W, 0.3, [para("三つの比較はいずれも同じ 3-stage 骨格・同じ学習設定。差分は「変更点」のみ。PSNR 変化は 10 シーン平均の差。", sz=10, color=GRAY)]))
cards = [
    ("サンプリング層のウェーブレットは有効", "+0.37 dB、しかもパラメータは増えない"),
    ("現在のブロック内帯域分割は有害", "−0.24 dB。同じ骨格で Base に負ける"),
    ("Mamba 置換だけでは微減", "−0.18 dB。Mamba の使い方に改善余地（次ページ以降）"),
]
cy = 2.85; cw = (W - 0.3) / 3; chh = 1.55
for i, (t, d) in enumerate(cards):
    cx = X0 + i * (cw + 0.15)
    sh.append(shape(cx, cy, cw, chh, PUR))
    sh.append(textbox(cx + 0.22, cy + 0.2, cw - 0.44, 0.3, [para(f"結論 {i+1}", sz=10, b=True, color="F3A6CC")]))
    sh.append(textbox(cx + 0.22, cy + 0.5, cw - 0.44, 0.6, [para(t, sz=13, b=True, color=WHITE, lnspc=110)]))
    sh.append(textbox(cx + 0.22, cy + 1.08, cw - 0.44, 0.42, [para(d, sz=10, color="E6D8F2", lnspc=108)]))
write_slide(5, "アブレーション：骨格を固定し 1 か所だけ置換", sh)

# ---------- slide 6: 帯域診断 ----------
LW = 4.15; RX = X0 + LW + 0.3; RW = W - LW - 0.3
sh = []
sh.append(textbox(X0, Y0, LW, 0.3, [para("① 帯域別誤差分解（Haar 2 段＋FFT リング）", sz=12, b=True, color=MID)]))
bul = [
    [("どのモデルも誤差は", {}), ("低域主体", {"b": True}), ("：LL2 が誤差エネルギーの約 42%、FFT 低域リングが約 75%。PSNR を決めるのは低域構造", {})],
    [("HyMCNetL3 vs SSR-S", {"b": True}), ("：LL2 のみ勝ち（+0.24 dB）、高域はすべて僅差で負け（約 −0.1 dB）。「Mamba が大域、畳み込みが細部」を支持する証拠", {})],
    [("V5 vs SSR-S", {"b": True}), ("：全帯域で 0.45〜0.69 dB 劣後。最悪は", {}), ("中域", {"b": True}), ("（2 段目サブバンドと 1 段目 HL / LH）、最高域 HH1 が最も差が小さい（−0.47）。高域分岐が弱いのではなく、分割後の各段の表現が全体的に劣化", {})],
]
sh.append(textbox(X0, Y0 + 0.36, LW, 2.6, [para(b, sz=10.5, bullet=True, spc_after=5, lnspc=110) for b in bul]))
sh.append(textbox(X0, 3.95, LW, 0.9, [para("HyMCNetL3 が SSR-S に勝つ 0.19 dB は 64×64 以下の低域に由来し、高域はむしろ僅かに劣る（右表②の HyMCNetL3 列で確認）。", sz=10, color=GRAY, lnspc=108)]))

sh.append(textbox(RX, Y0, RW, 0.3, [para("② サブバンド入替（oracle band-swap）PSNR", sz=12, b=True, color=MID)]))
rows = [["低域スケール", "V5 低域＋SSR-S 高域", "SSR-S 低域＋V5 高域", "HyMCNetL3 低域＋SSR-S 高域", "SSR-S 低域＋HyMCNetL3 高域"],
    ["128×128", "38.64", "38.79", "39.12", "38.92"],
    ["64×64", "38.81", "38.62", ("39.13", {"b": True, "fill": TINT}), "38.91"],
    ["32×32", "38.90", "38.53", "39.09", "38.95"],
    ["16×16", ("38.94", {"b": True, "fill": TINT}), "38.49", "39.04", "39.00"]]
tbl_y = Y0 + 0.36
sh.append(table(RX, tbl_y, [0.7, 0.8, 0.8, 0.975, RW - 3.275], rows, rowh=0.26, sz=9.5, header_h=0.46, header_sz=8))
tbl_y += 0.46 + 0.26 * 4
sh.append(textbox(RX, tbl_y + 0.03, RW, 0.22, [para("単体 PSNR：V5 38.51 ／ SSR-S 38.93 ／ HyMCNetL3 39.12", sz=8.5, color=GRAY)]))
sh.append(textbox(RX, tbl_y + 0.27, RW, 0.78,
                  [para([("V5 が SSR-S と並ぶのは 16×16 の最粗構造だけで、", {}), ("16〜128 の中低域はすべて劣る", {"b": True}), ("（各段のブロック内 Mamba が担当する LL そのもの）。一方 HyMCNetL3 は 64×64 低域を SSR-S の高域と組むだけで本体を上回り（39.13 > 39.12）、", {}), ("0.19 dB の優位は 64×64 以下の低域に由来", {"b": True}), ("、高域はむしろ僅かに劣る。", {})], sz=9.5, lnspc=106)]))
by = 3.92; bh = 1.2
sh.append(shape(RX, by, RW, bh, MAG))
sh.append(textbox(RX + 0.18, by + 0.12, RW - 0.36, 0.45, [para("結論：ブロック内低域分岐の Mamba が 2D 構造を学べていない", sz=11, b=True, color=WHITE, lnspc=110)]))
sh.append(textbox(RX + 0.18, by + 0.6, RW - 0.36, 0.55, [para("(a) 行優先の単方向 1D 走査では上下隣接画素が系列上 128 離れる　(b) 2 分岐がブロック内で互いに見えない", sz=9.5, color="FBE3EF", lnspc=108)]))
write_slide(6, "帯域診断：V5 が負けているのは低域側の Mamba", sh)

# ---------- slide 7: V6 ----------
sh = [textbox(X0, Y0, W, 0.5, [para([("4 方向走査（計算量 4 倍）は採らない。変更は低域分岐の Mamba の", {}), ("入力と走査順", {"b": True}), ("のみで、他は V5 と同一。", {})], sz=11, lnspc=110)])]
cards = [
    ("1　構造トークン", MAG, "4 サブバンドを 1 トークンに",
     "LL / HL / LH / HH を連結（4C）し 1×1 射影で C 次元に。各トークンが 2×2 近傍の「平均＋水平エッジ＋垂直エッジ＋対角」を符号化し、Mamba は孤立画素ではなく構造単位を走査。高域情報も見える。出力は LL にのみ加算。"),
    ("2　Hilbert 曲線走査", MID, "隣接トークンの 2D 距離を常に 1 に",
     "行優先では隣接トークン間の 2D 距離が平均約 126。Hilbert 順なら常に 1。パラメータ 0、固定置換 1 回のみ。"),
    ("3　2D 深さ方向畳み込み", PUR, "Mamba 内部の 1D 因果畳み込みを置換",
     "トークンが走査前に 2D 近傍を保持。状態空間パラメータは mamba_ssm の元の初期化をそのまま使用。"),
]
cy = 1.5; cw = (W - 0.3) / 3; chh = 2.3
for i, (lab, col, t, d) in enumerate(cards):
    cx = X0 + i * (cw + 0.15)
    sh.append(shape(cx, cy, cw, chh, CARD, line=LINE))
    sh.append(textbox(cx + 0.2, cy + 0.15, cw - 0.4, 0.28, [para(lab, sz=10.5, b=True, color=col)]))
    sh.append(textbox(cx + 0.2, cy + 0.45, cw - 0.4, 0.56, [para(t, sz=12.5, b=True, lnspc=110)]))
    sh.append(textbox(cx + 0.2, cy + 1.04, cw - 0.4, chh - 1.14, [para(d, sz=9.5, color="3A3550", lnspc=110)]))
tiles = [("パラメータ数", "3.15M"), ("初期偏差（SSRU は 0.116）", "0.072 ≈ 恒等写像"), ("1 step / VRAM", "0.57 s / 6.5 GB"), ("進捗", "10/15 開始、約 2 日")]
ty = 3.95; tw = (W - 0.45) / 4; th = 0.85
for i, (lab, val) in enumerate(tiles):
    tx = X0 + i * (tw + 0.15)
    sh.append(shape(tx, ty, tw, th, TINT))
    sh.append(textbox(tx + 0.15, ty + 0.1, tw - 0.3, 0.25, [para(lab, sz=9, color=GRAY)]))
    sh.append(textbox(tx + 0.15, ty + 0.38, tw - 0.3, 0.4, [para(val, sz=13, b=True)]))
sh.append(textbox(X0, 4.93, W, 0.25, [para("勾配は正常。300 epoch、学習設定は V5 と同一", sz=9, color=GRAY)]))
write_slide(7, "V6：符号化で 1D Mamba に 2D 構造を与える", sh)

# ---------- slide 3: 予定 ----------
LW = 5.3; RX = X0 + LW + 0.3; RW = W - LW - 0.3
plans = [
    ("V6 の評価とサブバンド入替の再実施", "同じ手順でテストし、ブロック内低域が HyMCNetL3 の低域に追いつくかを判定する"),
    ("追いつけば V6b、追いつけなければ V7", "V6b：V6 に「低域が高域を誘導する」ゲートを追加。V7：ブロック内分割を放棄し、ウェーブレットサンプリング＋Mamba ブロックを土台に Mamba をスペクトル次元（28 バンドの短い系列、CASSI の結合はスペクトル方向）で走査。空間構造は畳み込みと多重スケールに任せる"),
    ("論文用指標の再確認", "SSIM を MATLAB の cal_psnr_ssim.m で再計算し、論文と同じ基準に揃える"),
]
paras = []
for t, d in plans:
    paras.append(para(t, sz=11.5, b=True, bullet=True, spc_before=4))
    paras.append(para(d, sz=10.5, color="3A3550", bullet=False, spc_after=6, lnspc=110))
# indent descriptions to align with bullet text
paras = [p.replace('<a:pPr algn="l" marL="0" indent="0">', '<a:pPr algn="l" marL="228600" indent="0">') if "3A3550" in p else p for p in paras]
sh = [textbox(X0, Y0, LW, 3.9, paras)]
sh.append(textbox(X0, 4.93, LW, 0.25, [para("優先順位：1 を先に実施し、2 の分岐は 1 の結果で決める", sz=9, color=GRAY)]))
sh.append(shape(RX, Y0, RW, 3.3, PUR))
sh.append(textbox(RX + 0.2, Y0 + 0.15, RW - 0.4, 0.3, [para("リスク", sz=12, b=True, color="F3A6CC")]))
risks = [
    "ブロック内帯域分割は行き止まりの可能性。V6 が最後の検証で、以降は新規性をサンプリング層＋スペクトル走査に置く",
    "Mamba 系の学習-テスト差（約 3〜4 dB）が SSR（2.1 dB）より大きく、汎化に注意が必要",
    "テストセットは 10 シーンと小さく、0.1〜0.2 dB の差は慎重に解釈する",
]
rp = [para(r, sz=10, color="F3EAF8", bullet=True, spc_after=5, lnspc=110).replace(f'<a:buClr><a:srgbClr val="{MAG}"/>', '<a:buClr><a:srgbClr val="F3A6CC"/>') for r in risks]
sh.append(textbox(RX + 0.2, Y0 + 0.5, RW - 0.4, 2.7, rp))
write_slide(3, "予定とリスク", sh, title_old="予定", drop_text1=True)

# ---------- slide order ----------
pp = os.path.join(ROOT, "ppt", "presentation.xml")
s = open(pp, encoding="utf-8").read()
order = ['<p:sldId id="256" r:id="rId2"/>', '<p:sldId id="257" r:id="rId3"/>', '<p:sldId id="259" r:id="rId11"/>',
         '<p:sldId id="260" r:id="rId12"/>', '<p:sldId id="261" r:id="rId13"/>', '<p:sldId id="262" r:id="rId14"/>',
         '<p:sldId id="258" r:id="rId4"/>']
s = re.sub(r'<p:sldIdLst>.*?</p:sldIdLst>', '<p:sldIdLst>' + "".join(order) + '</p:sldIdLst>', s, flags=re.S)
open(pp, "w", encoding="utf-8").write(s)
print("done")
