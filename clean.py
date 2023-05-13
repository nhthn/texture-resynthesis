import pathlib

if __name__ == "__main__":
    in_files = pathlib.Path(__file__).parent / "in_files"
    for path in in_files.glob("*.resynthesized.wav"):
        path.unlink()
    for path in in_files.glob("*.truncated.wav"):
        path.unlink()
    for path in in_files.glob("*.info.json"):
        path.unlink()
