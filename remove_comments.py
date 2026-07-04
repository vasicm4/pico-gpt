
"""
Remove comments from Python files while preserving docstrings and code.
Usage: python remove_comments.py <file1.py> [file2.py ...]
"""
import sys
import tokenize
import os

def remove_comments_from_file(filepath):
    """Remove comments from a single Python file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    result = []
    last_lineno = -1
    last_col = 0

    try:
        tokens = list(tokenize.generate_tokens(iter(source.splitlines(keepends=True)).__next__))
    except tokenize.TokenError:

        with open(filepath, 'rb') as f:
            tokens = list(tokenize.tokenize(f.readline))

    for tok in tokens:
        token_type = tok.type
        token_string = tok.string
        start_line, start_col = tok.start
        end_line, end_col = tok.end


        if start_line > last_lineno:

            result.append('\n' * (start_line - last_lineno))
            last_col = 0
        elif start_col > last_col:

            result.append(' ' * (start_col - last_col))


        if token_type == tokenize.COMMENT:

            pass
        else:

            result.append(token_string)

        last_lineno, last_col = end_line, end_col


    new_source = ''.join(result)

    if source.endswith('\n') and not new_source.endswith('\n'):
        new_source += '\n'
    elif not source.endswith('\n') and new_source.endswith('\n'):
        new_source = new_source.rstrip('\n')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_source)

def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_comments.py <file1.py> [file2.py ...]")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}", file=sys.stderr)
            continue
        if not filepath.endswith('.py'):
            print(f"Skipping non-Python file: {filepath}", file=sys.stderr)
            continue
        print(f"Processing {filepath}")
        remove_comments_from_file(filepath)

if __name__ == '__main__':
    main()