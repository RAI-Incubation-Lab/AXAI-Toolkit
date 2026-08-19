# -*- coding: utf-8 -*-
"""读取 AXAI_Toolkit_Master_Proposal.docx 的纯文本内容（仅用 Python 标准库）。

用途：在不安装额外依赖的情况下，快速查看 Proposal 文档中的文字内容。

运行方式：
    python scripts/read_docx.py
"""
import re
import sys
import zipfile
from pathlib import Path


def extract_docx_text(docx_path: Path) -> str:
    """从 .docx 中提取正文文本。"""
    with zipfile.ZipFile(docx_path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8")
    # 段落结束转换为换行
    xml = xml.replace("</w:p>", "\n")
    # 去掉所有 XML 标签
    text = re.sub(r"<[^>]+>", "", xml)
    # 清理多余空行
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def main():
    docx_path = Path(__file__).resolve().parents[2] / "AXAI_Toolkit_Master_Proposal.docx"
    if not docx_path.exists():
        print(f"未找到文件: {docx_path}", file=sys.stderr)
        sys.exit(1)
    print(extract_docx_text(docx_path))


if __name__ == "__main__":
    main()
