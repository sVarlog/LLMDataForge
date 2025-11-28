#!/usr/bin/env python3
import json
from pathlib import Path

# Path to your placeholder file
PLACEHOLDER_PATH = Path("placeholder.txt")


def main() -> None:
	# 1. Read raw text
	raw = PLACEHOLDER_PATH.read_text(encoding="utf-8")

	# 2. Normalize newlines to '\n' (optional but safer across OSes)
	raw = raw.replace("\r\n", "\n").replace("\r", "\n")

	# 3. Escape as a JSON string
	#    - Escapes " -> \"
	#    - Escapes backslashes
	#    - Converts newline characters to \n
	#    Result includes surrounding quotes, e.g. "some text\nmore"
	escaped = json.dumps(raw, ensure_ascii=False)

	# 4. Write back to the same file (one line JSON-safe string)
	PLACEHOLDER_PATH.write_text(escaped, encoding="utf-8")


if __name__ == "__main__":
	main()
