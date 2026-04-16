from pathlib import Path
import re


def get_next_run_dir(base_output_dir: Path) -> Path:
    base_output_dir = Path(base_output_dir)
    existing = [
        d
        for d in base_output_dir.iterdir()
        if d.is_dir() and re.match(r"^run_\d+$", d.name)
    ]
    next_num = max((int(d.name.split("_")[1]) for d in existing), default=0) + 1
    run_dir = base_output_dir / f"run_{next_num:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_folder_from_num(run_num: int) -> Path:
    base_output_dir = Path(base_output_dir)
    existing = [
        d
        for d in base_output_dir.iterdir()
        if d.is_dir() and re.match(r"^run_\d+$", d.name)
    ]
    for dir in existing:
        if int(dir.lstrip("run_")) == run_num:
            return dir
    print("Не удалось найти запуск, возврат последнего запуска")
    return existing[-1]
