"""One local image-preparation entry point for trusted deployment profiles."""
from pathlib import Path
import subprocess
import sys

from sforge.pyromind.deployment import Deployment


def build_images(args):
    repo = Path(__file__).resolve().parents[2]
    deployment = Deployment.load(Path(args.deployment), repo)
    driver = deployment.path.with_name('build.py').resolve()
    if not driver.is_relative_to(repo / 'deploy' / 'pyromind') or not driver.is_file():
        raise ValueError('Deployment must have a reviewed build.py under deploy/pyromind')
    command = [sys.executable, str(driver)]
    if args.publish_audited:
        command.append('--publish-audited')
    elif not args.stage_only:
        command.append('--build')
    subprocess.run(command, cwd=repo, check=True)
