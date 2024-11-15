import subprocess
import sys

sys.path.append('workspace/WorldPeatland/code/')

geojson = '/workspace/WorldPeatland/sites/Norfolk.geojson'
site_name = 'Norfolk'

# Define the commands
commands = [
    # f'nohup python -m WorldPeatland.code.downloader_wp_test {geojson} /wp_data/sites/ > /workspace/logs/{site_name}_albedo_20241115.log &',
    f'python -m WorldPeatland.code.ts_generator_wp /wp_data/sites/{site_name}/ {geojson}',
    f'python -m WorldPeatland.code.apply_qa /wp_data/sites/{site_name}/ QA_settings',
    f'python -m WorldPeatland.code.smoothing /wp_data/sites/{site_name}/',
    f'python -m WorldPeatland.code.pixel_ts /wp_data/sites/{site_name}/ True',
    f'python -m WorldPeatland.code.pixel_ts /wp_data/sites/{site_name}/ False'
]

# Run each command sequentially
for cmd in commands:
    print(f'Running cmd: {cmd}')
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}: {cmd}")
        break
