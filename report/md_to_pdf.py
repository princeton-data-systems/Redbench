#!/usr/bin/env python3
"""Convert the markdown report to PDF with embedded images using fpdf2.

Handles: headings, bold, italic, code, tables with auto-sized columns,
images, bullet/numbered lists, and a table of contents rendered inline.
"""
from fpdf import FPDF
from pathlib import Path
import re

REPORT_MD = "report_redbench_workload_analysis.md"
REPORT_PDF = "report_redbench_workload_analysis.pdf"


def sanitize(text):
    """Replace unicode chars that latin-1 can't handle."""
    return (text
        .replace("\u2014", "--")
        .replace("\u2013", "-")
        .replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2026", "...").replace("\u2022", "-")
        .replace("\u00d7", "x").replace("\u2265", ">=")
        .replace("\u2264", "<=").replace("\u2260", "!=")
        .replace("\u2248", "~").replace("\u2192", "->")
        .replace("\u2713", "Y").replace("\u2717", "N")
        .replace("\u2714", "Y").replace("\u2718", "N")
        .replace("\u2610", "[ ]").replace("\u2611", "[x]")
        .replace("\u00b2", "^2").replace("\u00b3", "^3")
        .replace("\u2212", "-")
    )


def strip_markdown_links(text):
    """Convert [text](url) to just text."""
    return re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)


def md_inline_to_segments(text):
    """Parse inline markdown into segments of (style, text).

    Handles **bold**, *italic*, `code`, and plain text.
    Returns list of (style, text) tuples where style is a combo of B/I/C.
    """
    text = strip_markdown_links(text)
    text = sanitize(text)
    segments = []
    pos = 0
    pattern = re.compile(r'(\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`)')
    for m in pattern.finditer(text):
        if m.start() > pos:
            segments.append(("", text[pos:m.start()]))
        if m.group(2) is not None:
            segments.append(("B", m.group(2)))
        elif m.group(3) is not None:
            segments.append(("I", m.group(3)))
        elif m.group(4) is not None:
            segments.append(("C", m.group(4)))
        pos = m.end()
    if pos < len(text):
        segments.append(("", text[pos:]))
    if not segments:
        segments.append(("", text))
    return segments


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.toc_entries = []  # (level, title, page_no)

    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title, level=1):
        sizes = {1: 16, 2: 13, 3: 11}
        size = sizes.get(level, 11)
        clean_title = strip_markdown_links(sanitize(title))
        self.set_font("Helvetica", "B", size)
        self.set_text_color(40, 40, 40)
        if level == 1:
            self.ln(8)
        else:
            self.ln(5)
        self.toc_entries.append((level, clean_title, self.page_no()))
        self.multi_cell(0, size * 0.6, clean_title)
        if level == 1:
            self.set_draw_color(80)
            self.set_line_width(0.5)
            self.line(self.l_margin, self.get_y() + 1,
                      self.w - self.r_margin, self.get_y() + 1)
            self.ln(4)
        else:
            self.ln(3)

    def rich_text(self, segments, line_height=5.5):
        """Write a line of mixed bold/italic/plain text."""
        self.set_text_color(30, 30, 30)
        for style, text in segments:
            if style == "B":
                self.set_font("Helvetica", "B", 10)
            elif style == "I":
                self.set_font("Helvetica", "I", 10)
            elif style == "C":
                self.set_font("Courier", "", 9)
            else:
                self.set_font("Helvetica", "", 10)
            self.write(line_height, text)
        self.ln(line_height)

    def body_paragraph(self, text):
        """Render a paragraph with inline bold/italic/code."""
        segments = md_inline_to_segments(text)
        self.set_text_color(30, 30, 30)
        for style, txt in segments:
            if style == "B":
                self.set_font("Helvetica", "B", 10)
            elif style == "I":
                self.set_font("Helvetica", "I", 10)
            elif style == "C":
                self.set_font("Courier", "", 9)
            else:
                self.set_font("Helvetica", "", 10)
            self.write(5.5, txt)
        self.ln(5.5)
        self.ln(2)

    def add_image(self, path, caption=""):
        p = Path(path)
        if not p.exists():
            self.body_paragraph(f"[Image not found: {path}]")
            return
        if self.get_y() > 170:
            self.add_page()
        usable_w = self.w - self.l_margin - self.r_margin
        self.image(str(p), x=self.l_margin, w=usable_w)
        if caption:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100)
            self.multi_cell(0, 4, sanitize(caption))
        self.ln(3)

    def _measure_cell_height(self, text, col_width, font_name, font_style, font_size, cell_h):
        """Measure how tall a multi_cell would be without drawing it."""
        self.set_font(font_name, font_style, font_size)
        # Use get_string_width to estimate lines needed
        if col_width <= 2:
            return cell_h
        effective_w = col_width - 2  # padding
        text_w = self.get_string_width(text)
        if text_w <= effective_w:
            return cell_h
        n_lines = int(text_w / effective_w) + 1
        return max(cell_h, cell_h * n_lines)

    def add_table(self, headers, rows):
        """Render a table with auto-sized columns and proper row alignment."""
        usable_w = self.w - self.l_margin - self.r_margin
        n_cols = len(headers)
        font_size = 8
        cell_h = 5

        # Compute column widths based on content
        self.set_font("Helvetica", "B", font_size)
        col_max = []
        for j in range(n_cols):
            max_w = self.get_string_width(sanitize(headers[j].strip())) + 4
            for row in rows:
                if j < len(row):
                    cell_text = sanitize(row[j].strip())
                    w = self.get_string_width(cell_text) + 4
                    max_w = max(max_w, w)
            col_max.append(max_w)

        total = sum(col_max)
        if total > usable_w:
            min_w = 12
            col_widths = [max(min_w, c * usable_w / total) for c in col_max]
            s = sum(col_widths)
            col_widths = [c * usable_w / s for c in col_widths]
        else:
            col_widths = [c * usable_w / total for c in col_max]

        # Check if table fits on current page
        needed = cell_h * (1 + min(len(rows), 3))
        if self.get_y() + needed > 270:
            self.add_page()

        x_start = self.l_margin

        def draw_header():
            self.set_font("Helvetica", "B", font_size)
            self.set_fill_color(230, 230, 230)
            self.set_draw_color(180)
            y = self.get_y()
            for j, h in enumerate(headers):
                x = x_start + sum(col_widths[:j])
                self.rect(x, y, col_widths[j], cell_h)
                self.set_xy(x, y)
                self.cell(col_widths[j], cell_h, sanitize(h.strip()),
                          border=0, fill=True, align="C")
            self.set_y(y + cell_h)

        draw_header()

        # Rows
        self.set_font("Helvetica", "", font_size)
        for row in rows:
            cell_texts = []
            for j in range(n_cols):
                txt = sanitize(row[j].strip()) if j < len(row) else ""
                cell_texts.append(txt)

            # Pre-calculate row height
            row_h = cell_h
            for j, txt in enumerate(cell_texts):
                h = self._measure_cell_height(txt, col_widths[j],
                                               "Helvetica", "", font_size, cell_h)
                row_h = max(row_h, h)

            # Page break if needed
            if self.get_y() + row_h > 275:
                self.add_page()
                draw_header()
                self.set_font("Helvetica", "", font_size)

            y_before = self.get_y()

            # Draw cell borders first
            self.set_draw_color(180)
            for j in range(n_cols):
                x = x_start + sum(col_widths[:j])
                self.rect(x, y_before, col_widths[j], row_h)

            # Draw cell text
            for j, txt in enumerate(cell_texts):
                x = x_start + sum(col_widths[:j])
                self.set_font("Helvetica", "", font_size)
                self.set_xy(x, y_before)
                self.multi_cell(col_widths[j], cell_h, txt,
                                border=0, align="C")

            # Advance Y uniformly to bottom of row
            self.set_y(y_before + row_h)

        self.ln(3)


def parse_table(lines, start_idx):
    """Parse a markdown table starting at start_idx."""
    headers = [c.strip() for c in lines[start_idx].split("|") if c.strip()]
    rows = []
    idx = start_idx + 2  # skip separator
    while idx < len(lines) and "|" in lines[idx] and lines[idx].strip().startswith("|"):
        cells = [c.strip() for c in lines[idx].split("|") if c.strip()]
        rows.append(cells)
        idx += 1
    return headers, rows, idx


def is_toc_line(line):
    """Check if a line is a TOC entry like '1. [Executive Summary](#...)'."""
    return bool(re.match(r'^\s*\d+\.\s+\[', line.strip()) or
                re.match(r'^\s+- \[', line.strip()))


def render_report():
    md_text = Path(REPORT_MD).read_text()
    lines = md_text.split("\n")

    # --- PASS 1: render all content (skipping markdown TOC) to collect toc_entries ---
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Reserve page 1 for TOC -- start content on page 2
    # We'll render a title + placeholder on page 1, then content from page 2
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 15, sanitize("Redbench Workload Analysis"), align="C")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80)
    pdf.cell(0, 8, "Filter Diversity and Temporal Patterns", align="C")
    pdf.ln(8)
    pdf.ln(10)

    # We'll fill in the TOC after pass 1. For now, mark where TOC starts.
    toc_y_start = pdf.get_y()

    # Start content on a new page
    pdf.add_page()

    i = 0
    text_buffer = []
    in_toc = False

    def flush_text():
        nonlocal text_buffer
        if text_buffer:
            combined = " ".join(text_buffer)
            pdf.body_paragraph(combined)
            text_buffer = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip the markdown TOC section
        if stripped == "## Table of Contents":
            flush_text()
            in_toc = True
            i += 1
            continue
        if in_toc:
            if stripped == "---" or (stripped.startswith("## ") and stripped != "## Table of Contents"):
                in_toc = False
                if stripped == "---":
                    i += 1
                    continue
            else:
                i += 1
                continue

        # Skip the top-level title (already rendered on page 1)
        if stripped.startswith("# ") and not stripped.startswith("## "):
            i += 1
            continue

        # Horizontal rule
        if stripped == "---":
            flush_text()
            i += 1
            continue

        # Heading
        m = re.match(r'^(#{1,3})\s+(.*)', stripped)
        if m:
            flush_text()
            pdf.chapter_title(m.group(2), len(m.group(1)))
            i += 1
            continue

        # Image
        m = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', stripped)
        if m:
            flush_text()
            pdf.add_image(m.group(2), m.group(1))
            i += 1
            continue

        # Bold label line followed by image on next line
        m = re.match(r'^\*\*([^*]+)\*\*\s*(.*)', stripped)
        if m and not stripped.startswith("|"):
            if i + 1 < len(lines) and re.match(r'!\[', lines[i + 1].strip()):
                flush_text()
                label = sanitize(f"{m.group(1)} {m.group(2)}".strip())
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(30, 30, 30)
                pdf.multi_cell(0, 5.5, label)
                pdf.ln(1)
                i += 1
                continue

        # Table
        if (stripped.startswith("|") and i + 1 < len(lines)
                and re.match(r'^\|[\s\-:|]+\|', lines[i + 1].strip())):
            flush_text()
            headers, rows, end_idx = parse_table(lines, i)
            pdf.add_table(headers, rows)
            i = end_idx
            continue

        # Numbered list with bold label
        m = re.match(r'^(\d+)\.\s+\*\*(.+?)\*\*:\s*(.*)', stripped)
        if m:
            flush_text()
            num = m.group(1)
            label = sanitize(m.group(2))
            rest = sanitize(strip_markdown_links(m.group(3)))
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(8, 5.5, f"{num}.")
            pdf.set_font("Helvetica", "B", 10)
            pdf.write(5.5, f"{label}: ")
            pdf.set_font("Helvetica", "", 10)
            pdf.write(5.5, rest)
            pdf.ln(5.5)
            pdf.ln(1)
            i += 1
            continue

        # Numbered list plain
        m = re.match(r'^(\d+)\.\s+(.*)', stripped)
        if m:
            flush_text()
            num = m.group(1)
            content = m.group(2)
            segments = md_inline_to_segments(content)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(8, 5.5, f"{num}.")
            for style, txt in segments:
                if style == "B":
                    pdf.set_font("Helvetica", "B", 10)
                elif style == "I":
                    pdf.set_font("Helvetica", "I", 10)
                elif style == "C":
                    pdf.set_font("Courier", "", 9)
                else:
                    pdf.set_font("Helvetica", "", 10)
                pdf.write(5.5, txt)
            pdf.ln(5.5)
            pdf.ln(1)
            i += 1
            continue

        # Bullet point
        m = re.match(r'^[-*]\s+(.*)', stripped)
        if m:
            flush_text()
            content = m.group(1)
            segments = md_inline_to_segments(content)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            x_indent = pdf.l_margin + 6
            pdf.cell(6, 5.5, " -")
            old_l = pdf.l_margin
            pdf.set_left_margin(x_indent)
            pdf.set_x(x_indent)
            for style, txt in segments:
                if style == "B":
                    pdf.set_font("Helvetica", "B", 10)
                elif style == "I":
                    pdf.set_font("Helvetica", "I", 10)
                elif style == "C":
                    pdf.set_font("Courier", "", 9)
                else:
                    pdf.set_font("Helvetica", "", 10)
                pdf.write(5.5, txt)
            pdf.ln(5.5)
            pdf.set_left_margin(old_l)
            pdf.ln(1)
            i += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            flush_text()
            quote_text = sanitize(stripped.lstrip("> ").strip())
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(80, 80, 80)
            old_l = pdf.l_margin
            pdf.set_left_margin(old_l + 10)
            pdf.set_x(old_l + 10)
            pdf.multi_cell(0, 5.5, quote_text)
            pdf.set_left_margin(old_l)
            pdf.set_text_color(30, 30, 30)
            pdf.ln(2)
            i += 1
            continue

        # Empty line
        if not stripped:
            flush_text()
            i += 1
            continue

        # Regular text -- accumulate for paragraph
        text_buffer.append(stripped)
        i += 1

    flush_text()

    # --- PASS 2: render TOC on page 1 using collected toc_entries ---
    # Go back to page 1 and render the TOC
    pdf.page = 1
    pdf.set_y(toc_y_start)

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, "Table of Contents")
    pdf.ln(8)
    pdf.set_draw_color(80)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)

    for level, title, page in pdf.toc_entries:
        indent = (level - 1) * 8
        if level == 1:
            pdf.set_font("Helvetica", "B", 10)
        elif level == 2:
            pdf.set_font("Helvetica", "", 10)
        else:
            pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(30, 30, 30)
        pdf.set_x(pdf.l_margin + indent)
        page_str = str(page)
        title_w = pdf.w - pdf.l_margin - pdf.r_margin - indent - 15
        # Truncate title if too long for the available width
        display_title = sanitize(title)
        while pdf.get_string_width(display_title) > title_w - 2 and len(display_title) > 10:
            display_title = display_title[:-4] + "..."
        pdf.cell(title_w, 6, display_title)
        pdf.cell(15, 6, page_str, align="R")
        pdf.ln(6)

    pdf.output(REPORT_PDF)
    print(f"Written: {REPORT_PDF}")


if __name__ == "__main__":
    render_report()
