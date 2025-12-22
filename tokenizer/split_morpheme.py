import argparse
import re
from pathlib import Path
from typing import Sequence

from datasets import load_dataset
from mecab import MeCab

PosPair = tuple[str, str]

# 기본: 조사(J*), 어미(E*), 접사(X*)는 전부 "경계 제외"
DEFAULT_EXCLUDE_POS_PREFIXES = ("J", "E", "X")

# 추가로 경계 제외하고 싶은 "정확한 품사"
DEFAULT_EXCLUDE_POS_EXACT = {
    "VCP",  # 서술격 조사(이다)
    "VX",  # 보조용언
    # 기호류(문장부호 등)도 앞에 붙이도록 제외
    "SF",
    "SP",
    "SS",
    "SE",
    "SO",
    "SW",
}

# 참고: VV, VA 같은 내용어는 보통 제외하지 않습니다.
# 정말 제외하고 싶으면 DEFAULT_EXCLUDE_POS_EXACT에 "VV"를 추가하시면 됩니다.


def is_excluded_pos(
    pos: str,
    exclude_pos_prefixes: Sequence[str],
    exclude_pos_exact: set[str],
) -> bool:
    if pos in exclude_pos_exact:
        return True
    return any(pos.startswith(pfx) for pfx in exclude_pos_prefixes)


def chunk_by_pos(
    pairs: Sequence[PosPair],
    exclude_pos_prefixes: Sequence[str],
    exclude_pos_exact: set[str],
) -> list[str]:
    """
    - 기능 형태(제외 품사): 앞 덩어리에 붙인다
    - 내용 형태(그 외): 새 덩어리를 시작한다
    """
    chunks: list[str] = []
    cur = ""

    for morph, pos in pairs:
        if not is_excluded_pos(pos, exclude_pos_prefixes, exclude_pos_exact):
            # 기능 형태: 앞에 붙임
            cur = (cur + morph) if cur else morph
        else:
            # 내용 형태: 새 덩어리 시작
            if cur:
                chunks.append(cur)
            cur = morph

    if cur:
        chunks.append(cur)

    return chunks


def mecab_tag_file(
    input_path: Path,
    output_path: Path,
    tag: str = "<mecab>",
) -> None:
    mecab = MeCab()

    ds = load_dataset(input_path)
    data = [d["prompt"] for d in ds["train"]]

    with open(output_path, "w", encoding="utf-8") as out:
        for text in data:
            if not text:
                out.write("\n")
                continue

            # 형태소 리스트 추출
            morphs = mecab.morphs(text)
            if not morphs:
                out.write("\n")
                continue

            # 형태소 사이에 <mecab> 삽입
            out.write(mecab_tag_text(mecab, text, tag=tag) + "\n")


def mecab_tag_text(
    mecab: MeCab,
    text: str,
    tag: str = "<mecab>",
    exclude_pos_prefixes: Sequence[str] = DEFAULT_EXCLUDE_POS_PREFIXES,
    exclude_pos_exact: set[str] = DEFAULT_EXCLUDE_POS_EXACT,
) -> str:
    # 공백 보존: 공백 덩어리와 비공백 덩어리를 그대로 유지
    out_parts: list[str] = []
    for m in re.finditer(r"\s+|\S+", text):
        seg = m.group(0)
        if seg.isspace():
            out_parts.append(seg)
            continue

        pairs = mecab.pos(seg)
        if not pairs:
            out_parts.append(seg)
            continue

        chunks = chunk_by_pos(pairs, exclude_pos_prefixes, exclude_pos_exact)
        out_parts.append(tag.join(chunks) if chunks else "")

    return "".join(out_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_name", required=False, type=str, default="allganize/IFEval-Ko")
    parser.add_argument("--output", required=False, type=str, default="./assets/mecab_ko.txt")
    parser.add_argument("--tag", default="<mecab>")
    args = parser.parse_args()

    mecab_tag_file(input_path=args.input_name, output_path=args.output, tag=args.tag)


if __name__ == "__main__":
    main()
