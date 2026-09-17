# Build with .venv/bin/pyinstaller harness-usage.spec on the target macOS architecture.
from pathlib import Path

root = Path(SPECPATH)
package = root / 'src' / 'harness_usage'
a = Analysis(
    [str(package / '__main__.py')],
    pathex=[str(root / 'src')],
    binaries=[],
    datas=[(str(package / 'templates'), 'harness_usage/templates'),
           (str(package / 'static'), 'harness_usage/static'),
           (str(package / 'data'), 'harness_usage/data'),
           (str(package / 'schema.sql'), 'harness_usage'),
           (str(package / 'legacy_schema.sql'), 'harness_usage')],
    hiddenimports=['uvicorn.logging', 'uvicorn.loops.auto', 'uvicorn.protocols.http.auto',
                   'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan.on'],
    hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=[], noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='harness-usage',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='Harness Usage')
app = BUNDLE(coll, name='Harness Usage.app', icon=None,
             bundle_identifier='local.harness-usage',
             info_plist={'CFBundleDisplayName': 'Harness Usage', 'NSHighResolutionCapable': True})
