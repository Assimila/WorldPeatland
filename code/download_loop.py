import glob
import sys
sys.path.append('workspace/WorldPeatland/code/')
import subprocess


geojson = glob.glob(f'/workspace/WorldPeatland/sites/*.geojson')

geojson.sort()

for path in geojson:

    cmd = f'python downloader_sentinel.py {path} /data/sites'

    print(f'Running cmd: {cmd}')

    subprocess.run(cmd, shell= True)


